import os
import json
import time
import re

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

IMAGE_816 = "data/samples/invoice_816.jpg"
IMAGE_818 = "data/samples/invoice_818.jpg"

MAX_CROP_WIDTH = 1280
MAX_NEW_TOKENS = 128

# Start %, End %
CROP_REGIONS = [
    (0.58, 0.88),
    (0.62, 0.88),
    (0.65, 0.88),
]


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen2-VL + QLoRA...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    model,
    "outputs/qlora_mini"
)

model.eval()

device = next(model.parameters()).device

print("Model loaded.")
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
You are extracting financial totals from an invoice.

Carefully inspect the visible invoice section.

Find these exact values:

1. total_tax
   - Use the TOTAL TAX AMOUNT shown on the invoice.
   - If CGST and SGST are shown separately, their combined amount
     represents the total tax.
   - Do not use the taxable value as tax.

2. total_amount
   - Use the final GRAND TOTAL / TOTAL amount payable.
   - Do not use taxable value.
   - Do not use subtotal.
   - If a round-off value is shown, use the final total printed
     on the invoice.

Look carefully for:
- CGST
- SGST
- Tax Amount
- Total Tax Amount
- Total
- Grand Total
- Amount Chargeable
- round off

Return ONLY this JSON:

{
    "total_tax": null,
    "total_amount": null
}

Rules:
- Copy the numbers exactly as printed.
- Preserve decimal values.
- Do not invent values.
- If a value is genuinely not visible, use null.
- No explanation.
- No markdown.
"""


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text):

    if not text:
        return None

    text = text.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"```$",
            "",
            text
        )

        text = text.strip()

    try:

        data = json.loads(text)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        try:

            data = json.loads(
                text[start:end + 1]
            )

            if isinstance(data, dict):
                return data

        except Exception:
            pass

    return None


# ============================================================
# CREATE TARGETED CROP
# ============================================================

def create_crop(image, start_ratio, end_ratio):

    width, height = image.size

    start_y = int(height * start_ratio)
    end_y = int(height * end_ratio)

    crop = image.crop(
        (
            0,
            start_y,
            width,
            end_y
        )
    )

    # Resize while preserving aspect ratio.
    scale = MAX_CROP_WIDTH / crop.width

    new_width = int(crop.width * scale)
    new_height = int(crop.height * scale)

    crop = crop.resize(
        (new_width, new_height),
        Image.LANCZOS
    )

    return crop


# ============================================================
# EXTRACT TOTALS
# ============================================================

def extract_totals(
    image,
    start_ratio,
    end_ratio
):

    crop = create_crop(
        image,
        start_ratio,
        end_ratio
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": crop
                },
                {
                    "type": "text",
                    "text": PROMPT
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[crop],
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if hasattr(value, "to")
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True
        )

    elapsed = time.time() - start_time

    input_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[:, input_length:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    parsed = parse_json(output_text)

    peak_memory = None

    if torch.cuda.is_available():

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

    return {
        "start": start_ratio,
        "end": end_ratio,
        "crop_size": crop.size,
        "time": round(elapsed, 2),
        "peak_gpu": (
            round(peak_memory, 2)
            if peak_memory is not None
            else None
        ),
        "raw": output_text,
        "parsed": parsed
    }


# ============================================================
# TEST INVOICE
# ============================================================

def test_invoice(path, name):

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    if not os.path.exists(path):

        print("FILE NOT FOUND:")
        print(path)

        return

    image = Image.open(path).convert("RGB")

    print(
        "Original image:",
        image.size
    )

    for start, end in CROP_REGIONS:

        print("\n" + "-" * 70)

        print(
            f"Region: "
            f"{int(start * 100)}% → "
            f"{int(end * 100)}%"
        )

        result = extract_totals(
            image,
            start,
            end
        )

        print(
            "Crop size:",
            result["crop_size"]
        )

        print(
            "Time:",
            result["time"],
            "sec"
        )

        print(
            "Peak GPU:",
            result["peak_gpu"],
            "GB"
        )

        print("\nRAW OUTPUT:")
        print(result["raw"])

        print("\nPARSED:")

        if result["parsed"]:

            print(
                json.dumps(
                    result["parsed"],
                    indent=4
                )
            )

        else:

            print("INVALID JSON")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("PHASE 11.2 — TARGETED TOTALS CROP TEST")
    print("=" * 70)

    test_invoice(
        IMAGE_816,
        "INVOICE 816"
    )

    test_invoice(
        IMAGE_818,
        "INVOICE 818"
    )

    print("\n")
    print("=" * 70)
    print("TARGETED CROP TEST COMPLETE")
    print("=" * 70)