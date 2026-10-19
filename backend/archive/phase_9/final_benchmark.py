import os
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
# CONFIGURATION
# ============================================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "outputs/qlora_mini"

PROCESSED_GT = "data/processed/validation.jsonl"

NUM_SAMPLES = 20

MAX_IMAGE_WIDTH = 384
MAX_IMAGE_HEIGHT = 576

MAX_NEW_TOKENS = 256


# ============================================================
# GPU
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("PHASE 8.10D - FINAL 20-SAMPLE BENCHMARK")
print("=" * 70)

if torch.cuda.is_available():
    print("\nGPU:")
    print(torch.cuda.get_device_name(0))
else:
    print("\nWARNING: CUDA not available")


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower().strip()

    value = re.sub(r"\s+", " ", value)

    return value


def normalize_number(value):
    if value is None:
        return ""

    value = str(value)

    value = value.replace(",", "")
    value = value.replace("₹", "")
    value = value.replace("$", "")
    value = value.strip()

    return value


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading processed validation ground truth...")

gt_records = []

with open(PROCESSED_GT, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            gt_records.append(json.loads(line))

print("Processed validation records:", len(gt_records))


# ============================================================
# LOAD CORD
# ============================================================

print("\nLoading CORD validation dataset...")

cord = load_dataset(
    "naver-clova-ix/cord-v2",
    split="validation"
)

print("CORD validation images:", len(cord))


# ============================================================
# VERIFY ALIGNMENT
# ============================================================

print("\nChecking dataset alignment...")

assert len(gt_records) == len(cord), (
    f"Mismatch: GT={len(gt_records)}, CORD={len(cord)}"
)

print("Alignment confirmed.")
print("Processed JSONL index == CORD validation index")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen2-VL model...")

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

print("Base model loaded.")

# Load QLoRA adapter
model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

processor = AutoProcessor.from_pretrained(
    ADAPTER_PATH
)

print("QLoRA adapter loaded.")

print("Model device:", next(model.parameters()).device)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Extract invoice information from this image.

Return ONLY valid JSON using exactly this structure:

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

Rules:
- Extract only information visible in the invoice.
- If a field is missing, use null.
- Do not add explanations.
- Do not use markdown.
- Return only the JSON object.
"""


# ============================================================
# IMAGE RESIZE
# ============================================================

def resize_image(image):

    image = image.convert("RGB")

    width, height = image.size

    scale = min(
        MAX_IMAGE_WIDTH / width,
        MAX_IMAGE_HEIGHT / height,
        1.0
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )


# ============================================================
# PARSE JSON
# ============================================================

def extract_json(text):

    text = text.strip()

    try:
        return json.loads(text)
    except:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except:
            return None

    return None


# ============================================================
# LINE ITEM MATCHING
# ============================================================

def match_items(gt_items, pred_items):

    matched = []

    used = set()

    for gt in gt_items:

        gt_name = normalize_text(
            gt.get("name")
        )

        for i, pred in enumerate(pred_items):

            if i in used:
                continue

            pred_name = normalize_text(
                pred.get("name")
            )

            if (
                gt_name
                and pred_name
                and gt_name == pred_name
            ):
                matched.append((gt, pred))
                used.add(i)
                break

    return matched


# ============================================================
# METRICS
# ============================================================

json_correct = 0

total_correct = 0

name_correct = 0
quantity_correct = 0
price_correct = 0
complete_correct = 0

gt_item_count = 0
pred_item_count = 0
matched_item_count = 0

times = []

peak_vram = 0.0


# ============================================================
# INFERENCE LOOP
# ============================================================

print("\nStarting benchmark...")
print("Samples:", NUM_SAMPLES)
print("Resolution:", f"{MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}")
print("Max new tokens:", MAX_NEW_TOKENS)

print("\n" + "-" * 70)


for idx in range(NUM_SAMPLES):

    image = cord[idx]["image"]

    gt = gt_records[idx]

    image = resize_image(image)

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
        k: v.to(device)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

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

    times.append(elapsed)

    if torch.cuda.is_available():

        current_peak = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        peak_vram = max(
            peak_vram,
            current_peak
        )

    input_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[:, input_length:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    prediction = extract_json(output_text)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if prediction is not None:

        json_correct += 1

    # --------------------------------------------------------
    # GT ITEMS
    # --------------------------------------------------------

    target = gt.get("target", {})

    gt_items = target.get("line_items", [])

    if not isinstance(gt_items, list):
        gt_items = []

    if not isinstance(gt_items, list):
        gt_items = []

    gt_item_count += len(gt_items)

    # --------------------------------------------------------
    # PRED ITEMS
    # --------------------------------------------------------

    pred_items = []

    if prediction is not None:

        pred_items = prediction.get(
            "line_items",
            []
        )

        if not isinstance(pred_items, list):
            pred_items = []

    pred_item_count += len(pred_items)

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    if prediction is not None:

        gt_total = normalize_number(
            target.get("total_amount")
        )

        pred_total = normalize_number(
            prediction.get("total_amount")
        )

        if (
            gt_total
            and pred_total
            and gt_total == pred_total
        ):
            total_correct += 1

    # --------------------------------------------------------
    # MATCH ITEMS
    # --------------------------------------------------------

    matched = match_items(
        gt_items,
        pred_items
    )

    matched_item_count += len(matched)

    for gt_item, pred_item in matched:

        gt_name = normalize_text(
            gt_item.get("name")
        )

        pred_name = normalize_text(
            pred_item.get("name")
        )

        if gt_name and gt_name == pred_name:
            name_correct += 1

        gt_qty = normalize_number(
            gt_item.get("quantity")
        )

        pred_qty = normalize_number(
            pred_item.get("quantity")
        )

        if gt_qty and gt_qty == pred_qty:
            quantity_correct += 1

        gt_price = normalize_number(
            gt_item.get("price")
        )

        pred_price = normalize_number(
            pred_item.get("price")
        )

        if gt_price and gt_price == pred_price:
            price_correct += 1

        if (
            gt_name
            and gt_name == pred_name
            and gt_qty
            and gt_qty == pred_qty
            and gt_price
            and gt_price == pred_price
        ):
            complete_correct += 1

    print(
        f"Sample {idx + 1:02d}/{NUM_SAMPLES} | "
        f"{elapsed:.2f}s | "
        f"JSON={'YES' if prediction else 'NO'} | "
        f"GT items={len(gt_items)} | "
        f"Pred items={len(pred_items)} | "
        f"Matched={len(matched)}"
    )


# ============================================================
# FINAL METRICS
# ============================================================

json_accuracy = (
    json_correct / NUM_SAMPLES * 100
)

total_accuracy = (
    total_correct / NUM_SAMPLES * 100
)

if matched_item_count > 0:

    name_accuracy = (
        name_correct / matched_item_count * 100
    )

    quantity_accuracy = (
        quantity_correct / matched_item_count * 100
    )

    price_accuracy = (
        price_correct / matched_item_count * 100
    )

    complete_accuracy = (
        complete_correct / matched_item_count * 100
    )

else:

    name_accuracy = 0
    quantity_accuracy = 0
    price_accuracy = 0
    complete_accuracy = 0


precision = (
    matched_item_count / pred_item_count * 100
    if pred_item_count > 0
    else 0
)

recall = (
    matched_item_count / gt_item_count * 100
    if gt_item_count > 0
    else 0
)

if precision + recall > 0:

    f1 = (
        2 * precision * recall
        / (precision + recall)
    )

else:

    f1 = 0


avg_time = sum(times) / len(times)

fastest = min(times)

slowest = max(times)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 8.10D - FINAL 20-SAMPLE RESULTS")
print("=" * 70)

print(f"Configuration:     384x576 + 256 tokens")
print(f"Samples tested:    {NUM_SAMPLES}")

print()
print(f"JSON accuracy:     {json_accuracy:.2f}%")
print(f"Total accuracy:    {total_accuracy:.2f}%")

print()
print(f"GT items:          {gt_item_count}")
print(f"Predicted items:   {pred_item_count}")
print(f"Matched items:     {matched_item_count}")

print()
print(f"Name accuracy:     {name_accuracy:.2f}%")
print(f"Quantity accuracy: {quantity_accuracy:.2f}%")
print(f"Price accuracy:    {price_accuracy:.2f}%")
print(f"Complete item:     {complete_accuracy:.2f}%")

print()
print(f"Precision:         {precision:.2f}%")
print(f"Recall:            {recall:.2f}%")
print(f"F1 score:          {f1:.2f}%")

print()
print(f"Average inference: {avg_time:.2f} sec")
print(f"Fastest inference: {fastest:.2f} sec")
print(f"Slowest inference: {slowest:.2f} sec")

print()
print(f"Peak VRAM:         {peak_vram:.2f} GB")

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    print(f"Final allocated:   {allocated:.2f} GB")
    print(f"Final reserved:    {reserved:.2f} GB")

print("=" * 70)
print("PHASE 8.10D COMPLETE")
print("=" * 70)