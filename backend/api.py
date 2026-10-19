import io
import json
import re
import time

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "outputs/qlora_mini"

# Main invoice image
MAX_IMAGE_SIZE = 768

# First pass - complete invoice
MAX_NEW_TOKENS = 1024

# Second pass - bottom tax/total section
TOTALS_MAX_NEW_TOKENS = 128


# ============================================================
# PASS 1 PROMPT
# ============================================================

PROMPT = """
You are an invoice information extraction system.

Carefully inspect the ENTIRE invoice image from top to bottom.

Extract all visible invoice information into exactly the JSON structure
given below.

IMPORTANT RULES:

1. Extract EVERY visible line item from the invoice.
2. Preserve each item name as accurately as possible.
3. For every line item, extract quantity and price from the SAME row.
4. Do not invent information.
5. If a field is genuinely not visible, return null.

HEADER INFORMATION:

6. Extract:
   - vendor_name
   - invoice_number
   - invoice_date
   - gstin
   - state
   - state_code

LINE ITEMS:

7. Extract every visible line item.
8. Do not skip rows.
9. Quantity and price must belong to the same row.
10. Do not confuse quantity, rate and amount.

TOTALS:

11. Extract total_tax if clearly visible.
12. Extract total_amount if clearly visible.
13. total_tax means the complete tax charged on the invoice.
14. total_amount means the final payable/grand total.
15. Do not use taxable value as total_amount.
16. Do not use one individual CGST or SGST amount as total_tax.
17. If CGST and SGST are shown separately, total_tax is their combined
    amount.
18. Carefully inspect the bottom section of the invoice.

IMPORTANT:

19. Distinguish between:
    - taxable value
    - subtotal
    - CGST
    - SGST
    - total tax
    - round off
    - final total

20. Read commas, decimal points and minus signs carefully.
21. Return ONLY one valid JSON object.
22. Do NOT include explanations.
23. Do NOT use markdown.
24. Do NOT add text before or after the JSON.

Return exactly this structure:

{
  "vendor_name": null,
  "invoice_number": null,
  "invoice_date": null,
  "gstin": null,
  "state": null,
  "state_code": null,
  "total_tax": null,
  "total_amount": null,
  "line_items": [
    {
      "name": null,
      "quantity": null,
      "price": null
    }
  ]
}
"""


# ============================================================
# PASS 2 PROMPT
# ============================================================

TOTALS_PROMPT = """
You are an OCR system specialized in reading invoice totals.

Look ONLY at the invoice section shown in the image.

Your task is to read the exact numeric values printed in the
tax summary and final-total section.

Pay extremely close attention to small numbers, decimal points,
commas and labels.

Extract:

- cgst
- sgst
- total_tax
- taxable_value
- round_off
- total_amount

IMPORTANT:

1. Find the row labelled CGST and read its numeric amount.
2. Find the row labelled SGST and read its numeric amount.
3. total_tax is the combined tax amount.

4. If CGST and SGST are visible, calculate:

   total_tax = CGST + SGST

5. Find the final Total / Grand Total amount.
6. total_amount is the final payable amount.
7. Do NOT use taxable value as total_amount.
8. Do NOT use CGST alone as total_tax.
9. Do NOT use SGST alone as total_tax.
10. Do NOT estimate values.

For example:

CGST = 622.55
SGST = 622.55

Then:

total_tax = 1245.10

If the invoice shows:

Taxable Value = 24902.00
Round Off = -0.10
Final Total = 26147.00

Then:

total_amount = 26147.00

READ THE ACTUAL NUMBERS FROM THE IMAGE.

Return ONLY this JSON:

{
  "cgst": null,
  "sgst": null,
  "total_tax": null,
  "taxable_value": null,
  "round_off": null,
  "total_amount": null
}

Do not add explanations.
Do not use markdown.
"""


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Multi-Modal Edge AI - Invoice OCR API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL LOADING
# ============================================================

print("\n==============================================")
print("Loading Qwen2-VL + QLoRA model...")
print("==============================================")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

base_model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.eval()

print("Model loaded successfully.")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("==============================================\n")


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text):
    if not text:
        return None

    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"```$",
            "",
            text,
        )

        text = text.strip()

    # Direct JSON
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    # JSON embedded in text
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed

        except Exception:
            pass

    return None


# ============================================================
# IMAGE RESIZING
# ============================================================

def resize_image(image, max_size=MAX_IMAGE_SIZE):

    width, height = image.size

    longest_side = max(width, height)

    if longest_side <= max_size:
        return image

    scale = max_size / longest_side

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


# ============================================================
# CREATE BOTTOM IMAGE
# ============================================================

def create_bottom_crop(image):

    width, height = image.size

    # Focus on the lower portion of the invoice where
    # the tax summary and final total are located.
    crop_start = int(height * 0.62)

    bottom = image.crop(
        (
            0,
            crop_start,
            width,
            height,
        )
    )

    # IMPORTANT:
    # Do not aggressively shrink this crop.
    # The tax/total numbers are small and need more pixels.
    if bottom.width < 1024:

        scale = 1024 / bottom.width

        new_width = 1024
        new_height = int(bottom.height * scale)

        bottom = bottom.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

    return bottom


# ============================================================
# RUN QWEN EXTRACTION
# ============================================================

def run_qwen(image, prompt, max_new_tokens):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # --------------------------------------------------------
    # Chat template
    # --------------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # Move tensors to model device
    # --------------------------------------------------------

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        if torch.is_tensor(value)
        else value
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # --------------------------------------------------------
    # Remove input tokens
    # --------------------------------------------------------

    input_token_length = inputs["input_ids"].shape[1]

    generated_tokens = generated_ids[
        :,
        input_token_length:
    ]

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    output_text = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "model": MODEL_ID,
        "adapter": ADAPTER_PATH,
        "gpu_available": torch.cuda.is_available(),
    }


# ============================================================
# INVOICE ANALYSIS
# ============================================================

@app.post("/api/analyze")
async def analyze_invoice(
    document: UploadFile = File(...)
):

    start_time = time.time()

    try:

        # ====================================================
        # FILE VALIDATION
        # ====================================================

        if not document.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename provided.",
            )

        allowed_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        )

        filename_lower = document.filename.lower()

        if not filename_lower.endswith(
            allowed_extensions
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported file type. "
                    "Please upload JPG, JPEG, PNG or WEBP."
                ),
            )

        # ====================================================
        # READ FILE
        # ====================================================

        file_bytes = await document.read()

        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # ====================================================
        # OPEN IMAGE
        # ====================================================

        try:

            original_image = Image.open(
                io.BytesIO(file_bytes)
            ).convert("RGB")

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image.",
            )

        print("\n==============================================")
        print("NEW INVOICE REQUEST")
        print("==============================================")

        print(
            f"Filename: {document.filename}"
        )

        print(
            f"Original image size: {original_image.size}"
        )

        # ====================================================
        # CREATE TWO IMAGE VIEWS
        # ====================================================

        # Full invoice for header + line items
        full_image = resize_image(
            original_image,
            MAX_IMAGE_SIZE,
        )

        # Bottom crop for tax + total
        bottom_image = create_bottom_crop(
            original_image
        )

        print(
            f"Full image used: {full_image.size}"
        )

        print(
            f"Bottom crop used: {bottom_image.size}"
        )

        # ====================================================
        # GPU MEMORY RESET
        # ====================================================

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ====================================================
        # PASS 1
        # ====================================================

        print("\n----------------------------------------------")
        print("PASS 1: FULL INVOICE EXTRACTION")
        print("----------------------------------------------")

        pass1_output = run_qwen(
            full_image,
            PROMPT,
            MAX_NEW_TOKENS,
        )

        print("\n========== PASS 1 RAW OUTPUT ==========")
        print(pass1_output)
        print("=======================================\n")

        result = parse_json(
            pass1_output
        )

        if result is None:

            raise HTTPException(
                status_code=500,
                detail="Model returned invalid JSON during invoice extraction.",
            )

        # ====================================================
        # PASS 1 DEFAULT FIELDS
        # ====================================================

        expected_fields = [
            "vendor_name",
            "invoice_number",
            "invoice_date",
            "gstin",
            "state",
            "state_code",
            "total_tax",
            "total_amount",
            "line_items",
        ]

        for field in expected_fields:

            if field not in result:

                result[field] = (
                    []
                    if field == "line_items"
                    else None
                )

        # ====================================================
        # PASS 2
        # ====================================================

        print("\n----------------------------------------------")
        print("PASS 2: TAX + TOTAL EXTRACTION")
        print("----------------------------------------------")

        totals_output = run_qwen(
            bottom_image,
            TOTALS_PROMPT,
            TOTALS_MAX_NEW_TOKENS,
        )

        print("\n========== PASS 2 RAW OUTPUT ==========")
        print(totals_output)
        print("=======================================\n")

        totals = parse_json(
            totals_output
        )

        # ====================================================
        # MERGE PASS 2 RESULTS
        # ====================================================

        if totals is not None:

            print("\n========== TOTAL EXTRACTION ==========")

            print(
                "CGST:",
                totals.get("cgst")
            )

            print(
                "SGST:",
                totals.get("sgst")
            )

            print(
                "Total Tax:",
                totals.get("total_tax")
            )

            print(
                "Taxable Value:",
                totals.get("taxable_value")
            )

            print(
                "Round Off:",
                totals.get("round_off")
            )

            print(
                "Total Amount:",
                totals.get("total_amount")
            )

            print("======================================")

            # ------------------------------------------------
            # Prefer Pass 2 totals
            # ------------------------------------------------

            pass2_total_tax = totals.get(
                "total_tax"
            )

            pass2_total_amount = totals.get(
                "total_amount"
            )

            if pass2_total_tax not in (
                None,
                "",
                "null",
            ):

                result["total_tax"] = (
                    pass2_total_tax
                )

            if pass2_total_amount not in (
                None,
                "",
                "null",
            ):

                result["total_amount"] = (
                    pass2_total_amount
                )

            # ------------------------------------------------
            # Deterministic CGST + SGST fallback
            # ------------------------------------------------
            # Only used if the model extracted both individual
            # tax values but did not provide total_tax.

            if result.get("total_tax") in (
                None,
                "",
                "null",
            ):

                cgst = totals.get("cgst")
                sgst = totals.get("sgst")

                try:

                    if (
                        cgst is not None
                        and sgst is not None
                    ):

                        cgst_value = float(
                            str(cgst).replace(",", "")
                        )

                        sgst_value = float(
                            str(sgst).replace(",", "")
                        )

                        calculated_tax = (
                            cgst_value + sgst_value
                        )

                        result["total_tax"] = (
                            f"{calculated_tax:.2f}"
                        )

                        print(
                            "Calculated total tax from CGST + SGST:",
                            result["total_tax"],
                        )

                except Exception:

                    pass

        else:

            print(
                "WARNING: Pass 2 returned invalid JSON."
            )

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        if result.get("line_items") is None:

            result["line_items"] = []

        if not isinstance(
            result["line_items"],
            list,
        ):

            result["line_items"] = []

        # ====================================================
        # TIMING
        # ====================================================

        processing_time = (
            time.time() - start_time
        )

        # ====================================================
        # GPU MEMORY
        # ====================================================

        gpu_memory = None

        if torch.cuda.is_available():

            gpu_memory = {
                "allocated_gb": round(
                    torch.cuda.memory_allocated()
                    / (1024 ** 3),
                    2,
                ),
                "peak_gb": round(
                    torch.cuda.max_memory_allocated()
                    / (1024 ** 3),
                    2,
                ),
            }

        # ====================================================
        # FINAL LOG
        # ====================================================

        print("\n==============================================")
        print("FINAL EXTRACTION")
        print("==============================================")

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        print(
            f"\nCompleted in {processing_time:.2f}s"
        )

        if gpu_memory:

            print(
                f"Peak GPU memory: "
                f"{gpu_memory['peak_gb']:.2f} GB"
            )

        print("==============================================\n")

        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return {
            **result,

            "processing_time_sec": round(
                processing_time,
                2,
            ),

            "model": MODEL_ID,

            "adapter": ADAPTER_PATH,

            "gpu_memory": gpu_memory,

            "extraction_passes": 2,
        }

    # ========================================================
    # HTTP ERRORS
    # ========================================================

    except HTTPException:
        raise

    # ========================================================
    # UNEXPECTED ERRORS
    # ========================================================

    except torch.cuda.OutOfMemoryError:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail=(
                "GPU memory was insufficient to process "
                "this document."
            ),
        )

    except Exception as e:

        print("\n==============================================")
        print("ERROR")
        print("==============================================")

        print(
            str(e)
        )

        print("==============================================\n")

        raise HTTPException(
            status_code=500,
            detail=(
                "Invoice processing failed: "
                + str(e)
            ),
        )


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )