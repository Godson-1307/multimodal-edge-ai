import json
import re
import time
import os

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

OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "phase_9_1_error_analysis.json"
)


# ============================================================
# GPU
# ============================================================

print("=" * 70)
print("PHASE 9.1 - ERROR ANALYSIS")
print("=" * 70)

if torch.cuda.is_available():
    print("\nGPU:")
    print(torch.cuda.get_device_name(0))
else:
    print("\nWARNING: CUDA is not available")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading processed validation ground truth...")

gt_records = []

with open(PROCESSED_GT, "r", encoding="utf-8") as f:
    for line in f:
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
# CHECK ALIGNMENT
# ============================================================

print("\nChecking dataset alignment...")

for i in range(min(5, len(gt_records))):
    gt_id = gt_records[i]["image_id"]

    if gt_id != i:
        print(
            f"WARNING: image_id mismatch at index {i}: "
            f"image_id={gt_id}"
        )

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

processor = AutoProcessor.from_pretrained(
    MODEL_ID
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

print("Model loaded.")
print("Model device:", next(model.parameters()).device)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Extract the information from this receipt/invoice image.

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
- Extract every visible line item.
- Preserve the item name as accurately as possible.
- Extract quantity exactly as shown.
- Extract the corresponding price exactly as shown.
- Extract the final total amount if visible.
- Do not invent information.
- If a field is not visible, use null.
- Return only the JSON object.
"""


# ============================================================
# JSON PARSER
# ============================================================

def extract_json(text):

    text = text.strip()

    try:
        return json.loads(text)

    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:

        candidate = match.group(0)

        try:
            return json.loads(candidate)

        except Exception:
            return None

    return None


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value).lower().strip()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    value = re.sub(
        r"[^\w\s]",
        "",
        value
    )

    return value


def normalize_number(value):

    if value is None:
        return ""

    value = str(value)

    value = value.replace(",", "")
    value = value.replace("₹", "")
    value = value.replace("rs.", "")
    value = value.replace("rs", "")

    value = re.sub(
        r"[^\d.]",
        "",
        value
    )

    return value.strip()


# ============================================================
# ITEM MATCHING
# ============================================================

def match_items(gt_items, pred_items):

    matches = []

    used_predictions = set()

    for gt_index, gt_item in enumerate(gt_items):

        gt_name = normalize_text(
            gt_item.get("name")
        )

        best_match = None

        for pred_index, pred_item in enumerate(pred_items):

            if pred_index in used_predictions:
                continue

            pred_name = normalize_text(
                pred_item.get("name")
            )

            if gt_name and pred_name:

                if gt_name == pred_name:

                    best_match = pred_index
                    break

        if best_match is not None:

            used_predictions.add(best_match)

            matches.append(
                {
                    "gt_index": gt_index,
                    "pred_index": best_match
                }
            )

    return matches


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def analyze_errors(
    gt_items,
    pred_items,
    matches
):

    errors = []

    matched_gt = {
        x["gt_index"]
        for x in matches
    }

    matched_pred = {
        x["pred_index"]
        for x in matches
    }

    # --------------------------------------------------------
    # Missing items
    # --------------------------------------------------------

    for i, item in enumerate(gt_items):

        if i not in matched_gt:

            errors.append(
                {
                    "type": "missing_or_name_mismatch",
                    "ground_truth": item
                }
            )

    # --------------------------------------------------------
    # Extra predictions
    # --------------------------------------------------------

    for i, item in enumerate(pred_items):

        if i not in matched_pred:

            errors.append(
                {
                    "type": "extra_or_wrong_item",
                    "prediction": item
                }
            )

    # --------------------------------------------------------
    # Matched item field errors
    # --------------------------------------------------------

    for match in matches:

        gt = gt_items[
            match["gt_index"]
        ]

        pred = pred_items[
            match["pred_index"]
        ]

        gt_quantity = normalize_number(
            gt.get("quantity")
        )

        pred_quantity = normalize_number(
            pred.get("quantity")
        )

        gt_price = normalize_number(
            gt.get("price")
        )

        pred_price = normalize_number(
            pred.get("price")
        )

        if gt_quantity != pred_quantity:

            errors.append(
                {
                    "type": "quantity_error",
                    "ground_truth": gt,
                    "prediction": pred
                }
            )

        if gt_price != pred_price:

            errors.append(
                {
                    "type": "price_error",
                    "ground_truth": gt,
                    "prediction": pred
                }
            )

    return errors


# ============================================================
# RUN ANALYSIS
# ============================================================

results = []

error_counts = {
    "invalid_json": 0,
    "missing_or_name_mismatch": 0,
    "extra_or_wrong_item": 0,
    "quantity_error": 0,
    "price_error": 0,
    "total_error": 0,
    "correct": 0
}


print("\nStarting error analysis...")
print("Samples:", NUM_SAMPLES)
print(
    "Resolution:",
    f"{MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}"
)
print("Max new tokens:", MAX_NEW_TOKENS)

print("\n" + "-" * 70)


for i in range(NUM_SAMPLES):

    gt_record = gt_records[i]

    target = gt_record.get(
        "target",
        {}
    )

    gt_items = target.get(
        "line_items",
        []
    )

    gt_total = normalize_number(
        target.get("total_amount")
    )

    image = cord[i]["image"]

    # --------------------------------------------------------
    # Resize image
    # --------------------------------------------------------

    image = image.convert("RGB")

    image.thumbnail(
        (
            MAX_IMAGE_WIDTH,
            MAX_IMAGE_HEIGHT
        )
    )

    # --------------------------------------------------------
    # Prepare messages
    # --------------------------------------------------------

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
        k: v.to(model.device)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    input_length = inputs["input_ids"].shape[1]

    generated_only = generated_ids[
        :,
        input_length:
    ]

    output_text = processor.batch_decode(
        generated_only,
        skip_special_tokens=True
    )[0].strip()

    prediction = extract_json(
        output_text
    )

    # --------------------------------------------------------
    # JSON status
    # --------------------------------------------------------

    json_valid = prediction is not None

    if not json_valid:

        error_counts["invalid_json"] += 1

        result = {
            "sample": i + 1,
            "inference_time": round(
                elapsed,
                2
            ),
            "json_valid": False,
            "ground_truth": target,
            "raw_prediction": output_text,
            "prediction": None,
            "errors": [
                {
                    "type": "invalid_json"
                }
            ]
        }

        results.append(result)

        print(
            f"Sample {i + 1:02d}/{NUM_SAMPLES} | "
            f"{elapsed:.2f}s | "
            f"JSON=NO"
        )

        continue

    # --------------------------------------------------------
    # Prediction fields
    # --------------------------------------------------------

    pred_items = prediction.get(
        "line_items",
        []
    )

    pred_total = normalize_number(
        prediction.get("total_amount")
    )

    # --------------------------------------------------------
    # Match items
    # --------------------------------------------------------

    matches = match_items(
        gt_items,
        pred_items
    )

    # --------------------------------------------------------
    # Analyze errors
    # --------------------------------------------------------

    errors = analyze_errors(
        gt_items,
        pred_items,
        matches
    )

    # --------------------------------------------------------
    # Total error
    # --------------------------------------------------------

    if gt_total != pred_total:

        error_counts["total_error"] += 1

        errors.append(
            {
                "type": "total_error",
                "ground_truth": gt_total,
                "prediction": pred_total
            }
        )

    # --------------------------------------------------------
    # Count errors
    # --------------------------------------------------------

    for error in errors:

        error_type = error["type"]

        if error_type in error_counts:

            error_counts[
                error_type
            ] += 1

    if len(errors) == 0:

        error_counts["correct"] += 1

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "sample": i + 1,

        "inference_time": round(
            elapsed,
            2
        ),

        "json_valid": True,

        "ground_truth": target,

        "prediction": prediction,

        "raw_prediction": output_text,

        "gt_item_count": len(
            gt_items
        ),

        "pred_item_count": len(
            pred_items
        ),

        "matched_items": len(
            matches
        ),

        "errors": errors
    }

    results.append(result)

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        f"Sample {i + 1:02d}/{NUM_SAMPLES} | "
        f"{elapsed:.2f}s | "
        f"GT={len(gt_items)} | "
        f"Pred={len(pred_items)} | "
        f"Matched={len(matches)} | "
        f"Errors={len(errors)}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

report = {

    "configuration": {
        "model": MODEL_ID,
        "adapter": ADAPTER_PATH,
        "resolution": "384x576",
        "max_new_tokens": 256,
        "quantization": "4-bit NF4",
        "compute_dtype": "bfloat16",
        "use_cache": True,
        "inference_mode": True
    },

    "samples_tested": NUM_SAMPLES,

    "error_counts": error_counts,

    "samples": results
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 9.1 - ERROR ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nInvalid JSON:              "
    f"{error_counts['invalid_json']}"
)

print(
    f"Missing/name mismatch:     "
    f"{error_counts['missing_or_name_mismatch']}"
)

print(
    f"Extra/wrong item:          "
    f"{error_counts['extra_or_wrong_item']}"
)

print(
    f"Quantity errors:           "
    f"{error_counts['quantity_error']}"
)

print(
    f"Price errors:              "
    f"{error_counts['price_error']}"
)

print(
    f"Total errors:              "
    f"{error_counts['total_error']}"
)

print(
    f"Completely correct samples:"
    f" {error_counts['correct']}"
)

print(
    f"\nDetailed report saved to:"
    f"\n{OUTPUT_FILE}"
)

print("=" * 70)