import json
import time
import torch

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig
)
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "outputs/qlora_improved"

MAX_IMAGE_SIZE = 384
MAX_NEW_TOKENS = 256

VAL_JSON = "data/processed/validation.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
You are an invoice information extraction system.

Carefully inspect the ENTIRE invoice image and extract all visible
information into the JSON structure below.

IMPORTANT RULES:

1. Extract EVERY visible line item. Do not skip any item.
2. Preserve each product/item name as accurately as possible.
3. For every line item, identify its quantity and price from the
   SAME row whenever they are visible.
4. Do not invent information.
5. If a field is genuinely not visible, return null.
6. Carefully distinguish between:
   - subtotal
   - tax
   - total amount / grand total
7. For total_amount, use the final payable/grand total shown on
   the invoice, not the subtotal.
8. For total_tax, use the tax amount shown on the invoice.
9. Read numbers carefully, including commas and decimal points.
10. Return ONLY one valid JSON object.
11. Do NOT include explanations.
12. Do NOT use markdown.
13. Before returning the answer, check that all visible line items
    have been included.

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
# LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("PHASE 9.6 - IMPROVED ADAPTER DIAGNOSTIC")
print("=" * 70)

cord = load_dataset("naver-clova-ix/cord-v2")

with open(VAL_JSON, "r", encoding="utf-8") as f:
    validation_records = [json.loads(line) for line in f]

print("Validation records:", len(validation_records))


# ============================================================
# LOAD MODEL
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

print("\nLoading base model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Loading improved adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

print("Model + adapter loaded.")
print("Device:", DEVICE)


# ============================================================
# HELPER
# ============================================================

def resize_image(image, max_size):

    width, height = image.size

    scale = min(
        max_size / width,
        max_size / height
    )

    if scale >= 1:
        return image

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height))


# ============================================================
# TEST 3 SAMPLES
# ============================================================

for i in range(3):

    record = validation_records[i]

    image_id = record["image_id"]

    image = cord["validation"][image_id]["image"]

    image = resize_image(image, MAX_IMAGE_SIZE)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
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
        images=[image],
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    print("\n" + "=" * 70)
    print(f"SAMPLE {i + 1}")
    print("=" * 70)

    start = time.time()

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True
        )

    elapsed = time.time() - start

    input_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[:, input_length:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    print(f"Inference time: {elapsed:.2f} sec")

    print("\nRAW MODEL OUTPUT:")
    print("-" * 70)
    print(output_text)
    print("-" * 70)

    # Test JSON directly
    try:

        parsed = json.loads(output_text)

        print("\nJSON STATUS: VALID")

        print("\nParsed keys:")
        print(list(parsed.keys()))

    except Exception as e:

        print("\nJSON STATUS: INVALID")

        print("JSON error:")
        print(e)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)