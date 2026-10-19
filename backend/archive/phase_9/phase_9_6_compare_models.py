import os
import json
import time
import re
import torch

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

CURRENT_ADAPTER = "outputs/qlora_mini"
IMPROVED_ADAPTER = "outputs/qlora_improved"

VAL_JSON = "data/processed/validation.jsonl"

OUTPUT_FILE = "outputs/phase_9_6_model_comparison.json"

NUM_SAMPLES = 20

MAX_IMAGE_SIZE = 384
MAX_NEW_TOKENS = 256


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
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 9.6 - FINAL MODEL COMPARISON")
print("=" * 70)

print("\nLoading CORD dataset...")

cord = load_dataset("naver-clova-ix/cord-v2")

print("CORD validation samples:", len(cord["validation"]))


with open(VAL_JSON, "r", encoding="utf-8") as f:
    validation_records = [
        json.loads(line)
        for line in f
    ]

validation_records = validation_records[:NUM_SAMPLES]

print("Samples used:", len(validation_records))


# ============================================================
# LOAD VALIDATION GROUND TRUTH
# ============================================================

ground_truths = {}

for record in validation_records:

    image_id = record["image_id"]

    target = record.get("target", {})

    ground_truths[image_id] = target


# ============================================================
# IMAGE RESIZE
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

    return image.resize(
        (new_width, new_height)
    )


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text):

    if not text:
        return None

    text = text.strip()

    # Remove markdown fences if present
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

    # --------------------------------------------------------
    # Strict direct JSON parsing
    # --------------------------------------------------------

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

        return None

    except Exception:
        pass

    # --------------------------------------------------------
    # Try extracting the outermost JSON object
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Still invalid
    # --------------------------------------------------------

    return None


# ============================================================
# TEXT NORMALIZATION
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

    return value


# ============================================================
# NUMBER NORMALIZATION
# ============================================================

def normalize_number(value):

    if value is None:
        return ""

    value = str(value)

    # Remove commas and spaces
    value = value.replace(",", "")
    value = value.replace(" ", "")

    # Remove currency symbols
    value = re.sub(
        r"[^\d.\-]",
        "",
        value
    )

    # Remove trailing decimal zeros
    try:

        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(number)

    except Exception:

        return normalize_text(value)


# ============================================================
# GET LINE ITEMS
# ============================================================

def get_items(data):

    if not isinstance(data, dict):
        return []

    items = data.get(
        "line_items",
        []
    )

    if not isinstance(items, list):
        return []

    return items


# ============================================================
# MATCH LINE ITEMS
# ============================================================

def match_items(gt_items, pred_items):

    matched = []

    used_predictions = set()

    for gt in gt_items:

        gt_name = normalize_text(
            gt.get("name")
        )

        best_index = None

        # ----------------------------------------------------
        # Exact name matching
        # ----------------------------------------------------

        for index, pred in enumerate(pred_items):

            if index in used_predictions:
                continue

            pred_name = normalize_text(
                pred.get("name")
            )

            if (
                gt_name
                and pred_name
                and gt_name == pred_name
            ):
                best_index = index
                break

        # ----------------------------------------------------
        # Fallback partial matching
        # ----------------------------------------------------

        if best_index is None and gt_name:

            for index, pred in enumerate(pred_items):

                if index in used_predictions:
                    continue

                pred_name = normalize_text(
                    pred.get("name")
                )

                if not pred_name:
                    continue

                if (
                    gt_name in pred_name
                    or pred_name in gt_name
                ):
                    best_index = index
                    break

        # ----------------------------------------------------
        # Record match
        # ----------------------------------------------------

        if best_index is not None:

            used_predictions.add(
                best_index
            )

            matched.append(
                (
                    gt,
                    pred_items[best_index]
                )
            )

    return matched


# ============================================================
# EVALUATE ONE PREDICTION
# ============================================================

def evaluate_prediction(gt, prediction):

    gt_items = get_items(gt)

    if prediction is None:

        return {
            "valid_json": False,
            "total_correct": False,
            "gt_items": len(gt_items),
            "pred_items": 0,
            "matched_items": 0,
            "quantity_correct": 0,
            "price_correct": 0,
            "complete_items": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    pred_items = get_items(prediction)

    matched = match_items(
        gt_items,
        pred_items
    )

    matched_count = len(matched)

    # --------------------------------------------------------
    # Quantity / price
    # --------------------------------------------------------

    quantity_correct = 0
    price_correct = 0
    complete_items = 0

    for gt_item, pred_item in matched:

        qty_ok = (
            normalize_number(
                gt_item.get("quantity")
            )
            ==
            normalize_number(
                pred_item.get("quantity")
            )
        )

        price_ok = (
            normalize_number(
                gt_item.get("price")
            )
            ==
            normalize_number(
                pred_item.get("price")
            )
        )

        if qty_ok:
            quantity_correct += 1

        if price_ok:
            price_correct += 1

        if qty_ok and price_ok:
            complete_items += 1

    # --------------------------------------------------------
    # Precision / Recall / F1
    # --------------------------------------------------------

    if len(pred_items) > 0:

        precision = (
            matched_count
            / len(pred_items)
        ) * 100

    else:

        precision = 0.0

    if len(gt_items) > 0:

        recall = (
            matched_count
            / len(gt_items)
        ) * 100

    else:

        recall = 0.0

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    # --------------------------------------------------------
    # Total comparison
    # --------------------------------------------------------

    gt_total = normalize_number(
        gt.get("total_amount")
    )

    pred_total = normalize_number(
        prediction.get("total_amount")
    )

    total_correct = (
        gt_total != ""
        and pred_total != ""
        and gt_total == pred_total
    )

    return {
        "valid_json": True,
        "total_correct": total_correct,
        "gt_items": len(gt_items),
        "pred_items": len(pred_items),
        "matched_items": matched_count,
        "quantity_correct": quantity_correct,
        "price_correct": price_correct,
        "complete_items": complete_items,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(adapter_path=None):

    print("\n" + "-" * 70)

    if adapter_path is None:

        print("Loading BASE model")

    else:

        print(
            "Loading model + adapter:"
        )

        print(adapter_path)

    print("-" * 70)

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

    if adapter_path is not None:

        model = PeftModel.from_pretrained(
            model,
            adapter_path
        )

    model.eval()

    return model, processor


# ============================================================
# RUN MODEL
# ============================================================

def run_model(model, processor):

    results = []

    total_time = 0.0

    peak_memory = 0.0

    for index, record in enumerate(
        validation_records
    ):

        image_id = record["image_id"]

        image = cord[
            "validation"
        ][image_id]["image"]

        image = resize_image(
            image,
            MAX_IMAGE_SIZE
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

            return_tensors="pt"
        )

        inputs = {

            key: value.to(model.device)
            if torch.is_tensor(value)
            else value

            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Reset GPU peak statistics
        # ----------------------------------------------------

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        with torch.inference_mode():

            output_ids = model.generate(

                **inputs,

                max_new_tokens=MAX_NEW_TOKENS,

                do_sample=False,

                use_cache=True
            )

        elapsed = (
            time.time()
            - start_time
        )

        total_time += elapsed

        # ----------------------------------------------------
        # Decode only generated tokens
        # ----------------------------------------------------

        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated_ids = (
            output_ids[:, input_length:]
        )

        output_text = processor.batch_decode(

            generated_ids,

            skip_special_tokens=True
        )[0].strip()

        prediction = parse_json(
            output_text
        )

        gt = ground_truths[image_id]

        metrics = evaluate_prediction(
            gt,
            prediction
        )

        if torch.cuda.is_available():

            current_peak = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )

            peak_memory = max(
                peak_memory,
                current_peak
            )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        if prediction is None:

            print(
                f"Sample {index + 1:02d} | "
                f"{elapsed:.2f}s | "
                f"INVALID JSON"
            )

        else:

            print(
                f"Sample {index + 1:02d} | "
                f"{elapsed:.2f}s | "
                f"GT={metrics['gt_items']} | "
                f"Pred={metrics['pred_items']} | "
                f"Matched={metrics['matched_items']}"
            )

        # ----------------------------------------------------
        # Store detailed result
        # ----------------------------------------------------

        results.append({

            "sample": index + 1,

            "image_id": image_id,

            "inference_time": elapsed,

            "valid_json": metrics[
                "valid_json"
            ],

            "metrics": metrics,

            "raw_output": output_text,

            "prediction": prediction
        })

    # ========================================================
    # AGGREGATE
    # ========================================================

    total_samples = len(results)

    valid_json_count = sum(

        1

        for result in results

        if result["metrics"]["valid_json"]
    )

    total_correct_count = sum(

        1

        for result in results

        if result["metrics"]["total_correct"]
    )

    gt_items = sum(

        result["metrics"]["gt_items"]

        for result in results
    )

    pred_items = sum(

        result["metrics"]["pred_items"]

        for result in results
    )

    matched_items = sum(

        result["metrics"]["matched_items"]

        for result in results
    )

    quantity_correct = sum(

        result["metrics"]["quantity_correct"]

        for result in results
    )

    price_correct = sum(

        result["metrics"]["price_correct"]

        for result in results
    )

    complete_items = sum(

        result["metrics"]["complete_items"]

        for result in results
    )

    # --------------------------------------------------------
    # Aggregate precision / recall / F1
    # --------------------------------------------------------

    if pred_items > 0:

        precision = (
            matched_items
            / pred_items
        ) * 100

    else:

        precision = 0.0

    if gt_items > 0:

        recall = (
            matched_items
            / gt_items
        ) * 100

    else:

        recall = 0.0

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    # --------------------------------------------------------
    # Accuracy percentages
    # --------------------------------------------------------

    json_accuracy = (
        valid_json_count
        / total_samples
    ) * 100

    total_accuracy = (
        total_correct_count
        / total_samples
    ) * 100

    if matched_items > 0:

        quantity_accuracy = (
            quantity_correct
            / matched_items
        ) * 100

        price_accuracy = (
            price_correct
            / matched_items
        ) * 100

        complete_accuracy = (
            complete_items
            / matched_items
        ) * 100

    else:

        quantity_accuracy = 0.0
        price_accuracy = 0.0
        complete_accuracy = 0.0

    average_time = (
        total_time
        / total_samples
    )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    summary = {

        "samples_tested":
            total_samples,

        "json_accuracy":
            json_accuracy,

        "total_accuracy":
            total_accuracy,

        "gt_items":
            gt_items,

        "predicted_items":
            pred_items,

        "matched_items":
            matched_items,

        "quantity_accuracy":
            quantity_accuracy,

        "price_accuracy":
            price_accuracy,

        "complete_item_accuracy":
            complete_accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "average_inference_seconds":
            average_time,

        "peak_gpu_memory_gb":
            peak_memory
    }

    return summary, results


# ============================================================
# MAIN
# ============================================================

all_results = {}


# ============================================================
# 1. BASE MODEL
# ============================================================

print("\n\n")
print("=" * 70)
print("MODEL 1/3 - BASE QWEN2-VL")
print("=" * 70)

model, processor = load_model(
    adapter_path=None
)

base_summary, base_details = run_model(
    model,
    processor
)

all_results["base"] = {

    "model":
        "Qwen2-VL-2B-Instruct",

    "adapter":
        None,

    "summary":
        base_summary,

    "details":
        base_details
}

del model
del processor

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 2. CURRENT QLORA
# ============================================================

print("\n\n")
print("=" * 70)
print("MODEL 2/3 - CURRENT QLORA")
print("=" * 70)

model, processor = load_model(
    adapter_path=CURRENT_ADAPTER
)

current_summary, current_details = run_model(
    model,
    processor
)

all_results["current_qlora"] = {

    "model":
        "Qwen2-VL-2B-Instruct",

    "adapter":
        CURRENT_ADAPTER,

    "summary":
        current_summary,

    "details":
        current_details
}

del model
del processor

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 3. IMPROVED QLORA
# ============================================================

print("\n\n")
print("=" * 70)
print("MODEL 3/3 - IMPROVED QLORA")
print("=" * 70)

model, processor = load_model(
    adapter_path=IMPROVED_ADAPTER
)

improved_summary, improved_details = run_model(
    model,
    processor
)

all_results["improved_qlora"] = {

    "model":
        "Qwen2-VL-2B-Instruct",

    "adapter":
        IMPROVED_ADAPTER,

    "summary":
        improved_summary,

    "details":
        improved_details
}

del model
del processor

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

print("\n\n")
print("=" * 100)
print("PHASE 9.6 - FINAL MODEL COMPARISON")
print("=" * 100)

print(
    f"{'Metric':<30}"
    f"{'BASE':>15}"
    f"{'CURRENT':>15}"
    f"{'IMPROVED':>15}"
)

print("-" * 100)


def show_metric(
    name,
    key,
    suffix="%"
):

    base = all_results[
        "base"
    ]["summary"][key]

    current = all_results[
        "current_qlora"
    ]["summary"][key]

    improved = all_results[
        "improved_qlora"
    ]["summary"][key]

    print(
        f"{name:<30}"
        f"{base:>14.2f}{suffix}"
        f"{current:>14.2f}{suffix}"
        f"{improved:>14.2f}{suffix}"
    )


show_metric(
    "JSON Accuracy",
    "json_accuracy"
)

show_metric(
    "Total Accuracy",
    "total_accuracy"
)

show_metric(
    "Quantity Accuracy",
    "quantity_accuracy"
)

show_metric(
    "Price Accuracy",
    "price_accuracy"
)

show_metric(
    "Complete Item Accuracy",
    "complete_item_accuracy"
)

show_metric(
    "Precision",
    "precision"
)

show_metric(
    "Recall",
    "recall"
)

show_metric(
    "F1 Score",
    "f1"
)

print("-" * 100)

show_metric(
    "Average Inference Time",
    "average_inference_seconds",
    " sec"
)

show_metric(
    "Peak GPU Memory",
    "peak_gpu_memory_gb",
    " GB"
)

print("-" * 100)

print(
    f"{'GT Items':<30}"
    f"{all_results['base']['summary']['gt_items']:>15}"
    f"{all_results['current_qlora']['summary']['gt_items']:>15}"
    f"{all_results['improved_qlora']['summary']['gt_items']:>15}"
)

print(
    f"{'Predicted Items':<30}"
    f"{all_results['base']['summary']['predicted_items']:>15}"
    f"{all_results['current_qlora']['summary']['predicted_items']:>15}"
    f"{all_results['improved_qlora']['summary']['predicted_items']:>15}"
)

print(
    f"{'Matched Items':<30}"
    f"{all_results['base']['summary']['matched_items']:>15}"
    f"{all_results['current_qlora']['summary']['matched_items']:>15}"
    f"{all_results['improved_qlora']['summary']['matched_items']:>15}"
)


# ============================================================
# BEST MODEL
# ============================================================

base_f1 = all_results[
    "base"
]["summary"]["f1"]

current_f1 = all_results[
    "current_qlora"
]["summary"]["f1"]

improved_f1 = all_results[
    "improved_qlora"
]["summary"]["f1"]


scores = {

    "BASE":
        base_f1,

    "CURRENT QLORA":
        current_f1,

    "IMPROVED QLORA":
        improved_f1
}

best_model = max(
    scores,
    key=scores.get
)


print("\n")
print("=" * 70)
print("BEST MODEL BY LINE-ITEM F1")
print("=" * 70)

print(
    f"{best_model}: "
    f"{scores[best_model]:.2f}% F1"
)


print("\n")
print("=" * 70)
print("RESULT FILE")
print("=" * 70)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print("\n")
print("=" * 70)
print("PHASE 9.6 COMPLETE")
print("=" * 70)