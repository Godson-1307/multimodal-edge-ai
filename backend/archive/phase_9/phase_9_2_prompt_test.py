import json
import re
import time
import torch

from PIL import Image
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
ADAPTER_PATH = "outputs/qlora_mini"

NUM_SAMPLES = 20

MAX_IMAGE_WIDTH = 384
MAX_IMAGE_HEIGHT = 576
MAX_NEW_TOKENS = 256


# ============================================================
# IMPROVED PROMPT
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
# HELPERS
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-z0-9]+", "", value)

    return value


def normalize_number(value):
    if value is None:
        return ""

    value = str(value).strip()

    # Remove common formatting characters
    value = value.replace(",", "")
    value = value.replace("₹", "")
    value = value.replace("$", "")
    value = value.strip()

    return value


def parse_json(text):

    text = text.strip()

    # Remove accidental markdown
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def match_items(gt_items, pred_items):

    matches = []

    used_predictions = set()

    for gt in gt_items:

        gt_name = normalize_text(gt.get("name"))

        best_index = None

        for i, pred in enumerate(pred_items):

            if i in used_predictions:
                continue

            pred_name = normalize_text(pred.get("name"))

            if gt_name and pred_name and gt_name == pred_name:
                best_index = i
                break

        if best_index is not None:
            used_predictions.add(best_index)

            matches.append(
                (
                    gt,
                    pred_items[best_index]
                )
            )

    return matches


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("PHASE 9.2 - IMPROVED PROMPT TEST")
print("=" * 70)

print("\nLoading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Model loaded.")
print("Device:", next(model.parameters()).device)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading CORD dataset...")

dataset = load_dataset(
    "naver-clova-ix/cord-v2"
)

validation = dataset["validation"]

with open(
    "data/processed/validation.jsonl",
    "r",
    encoding="utf-8"
) as f:
    processed_records = [
        json.loads(line)
        for line in f
    ]

print("Validation samples:", len(validation))


# ============================================================
# METRICS
# ============================================================

valid_json = 0

gt_total_count = 0
correct_total = 0

gt_items_count = 0
pred_items_count = 0
matched_items_count = 0

quantity_correct = 0
price_correct = 0
complete_correct = 0

total_time = 0
peak_vram = 0


# ============================================================
# INFERENCE
# ============================================================

for idx in range(NUM_SAMPLES):

    image = validation[idx]["image"]

    # Resize while preserving aspect ratio
    image.thumbnail(
        (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)
    )

    gt = processed_records[idx]["target"]

    gt_items = gt.get("line_items", [])
    gt_total = normalize_number(
        gt.get("total_amount")
    )

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
        padding=True,
        return_tensors="pt"
    )

    inputs = {
        k: v.to("cuda") if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    torch.cuda.reset_peak_memory_stats()

    start = time.time()

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True
        )

    elapsed = time.time() - start

    total_time += elapsed

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    peak_vram = max(
        peak_vram,
        peak
    )

    # Remove prompt tokens
    input_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[:, input_length:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    prediction = parse_json(output_text)

    if prediction is not None:

        valid_json += 1

    else:

        prediction = {
            "line_items": []
        }

    pred_items = prediction.get(
        "line_items",
        []
    )

    pred_total = normalize_number(
        prediction.get("total_amount")
    )

    # Total
    gt_total_count += 1

    if gt_total and pred_total:
        if gt_total == pred_total:
            correct_total += 1

    # Line items
    gt_items_count += len(gt_items)
    pred_items_count += len(pred_items)

    matches = match_items(
        gt_items,
        pred_items
    )

    matched_items_count += len(matches)

    sample_errors = 0

    # Check item fields
    for gt_item, pred_item in matches:

        gt_qty = normalize_number(
            gt_item.get("quantity")
        )

        pred_qty = normalize_number(
            pred_item.get("quantity")
        )

        gt_price = normalize_number(
            gt_item.get("price")
        )

        pred_price = normalize_number(
            pred_item.get("price")
        )

        if gt_qty == pred_qty:
            quantity_correct += 1
        else:
            sample_errors += 1

        if gt_price == pred_price:
            price_correct += 1
        else:
            sample_errors += 1

        if (
            gt_qty == pred_qty
            and
            gt_price == pred_price
        ):
            complete_correct += 1

    # Missing / unmatched items
    unmatched = len(gt_items) - len(matches)

    sample_errors += unmatched

    # Total error
    if gt_total and pred_total != gt_total:
        sample_errors += 1

    print(
        f"Sample {idx+1:02d}/{NUM_SAMPLES} | "
        f"{elapsed:.2f}s | "
        f"GT={len(gt_items)} | "
        f"Pred={len(pred_items)} | "
        f"Matched={len(matches)} | "
        f"Errors={sample_errors}"
    )


# ============================================================
# FINAL METRICS
# ============================================================

precision = (
    matched_items_count / pred_items_count * 100
    if pred_items_count
    else 0
)

recall = (
    matched_items_count / gt_items_count * 100
    if gt_items_count
    else 0
)

if precision + recall > 0:
    f1 = (
        2 * precision * recall
        / (precision + recall)
    )
else:
    f1 = 0


quantity_accuracy = (
    quantity_correct / matched_items_count * 100
    if matched_items_count
    else 0
)

price_accuracy = (
    price_correct / matched_items_count * 100
    if matched_items_count
    else 0
)

complete_accuracy = (
    complete_correct / matched_items_count * 100
    if matched_items_count
    else 0
)

total_accuracy = (
    correct_total / gt_total_count * 100
    if gt_total_count
    else 0
)

avg_time = total_time / NUM_SAMPLES


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 9.2 - IMPROVED PROMPT RESULTS")
print("=" * 70)

print(f"Samples tested:       {NUM_SAMPLES}")
print(f"JSON accuracy:        {valid_json / NUM_SAMPLES * 100:.2f}%")

print(f"Total accuracy:       {total_accuracy:.2f}%")

print()
print(f"GT items:             {gt_items_count}")
print(f"Predicted items:      {pred_items_count}")
print(f"Matched items:        {matched_items_count}")

print()
print(f"Quantity accuracy:    {quantity_accuracy:.2f}%")
print(f"Price accuracy:       {price_accuracy:.2f}%")
print(f"Complete item:        {complete_accuracy:.2f}%")

print()
print(f"Precision:            {precision:.2f}%")
print(f"Recall:               {recall:.2f}%")
print(f"F1 score:             {f1:.2f}%")

print()
print(f"Average inference:    {avg_time:.2f} sec")
print(f"Peak VRAM:            {peak_vram:.2f} GB")

print("=" * 70)
print("PHASE 9.2 COMPLETE")
print("=" * 70)