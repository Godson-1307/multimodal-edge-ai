import json
import time
import re
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

VALIDATION_JSONL = "data/processed/validation.jsonl"

NUM_SAMPLES = 5

# Configurations to compare
CONFIGS = [
    ("384x576_128", 384, 576, 128),
    ("384x576_256", 384, 576, 256),

    ("640x960_128", 640, 960, 128),
    ("640x960_256", 640, 960, 256),

    ("768x1152_128", 768, 1152, 128),
    ("768x1152_256", 768, 1152, 256),
]


# ============================================================
# GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available.")

device = torch.device("cuda:0")

print("=" * 70)
print("PHASE 8.10C - FINAL CONFIGURATION SELECTION")
print("=" * 70)

print("\nGPU:")
print(torch.cuda.get_device_name(0))


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading processed validation ground truth...")

ground_truth_records = []

with open(VALIDATION_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        ground_truth_records.append(json.loads(line))

print("Processed validation records:", len(ground_truth_records))


# ============================================================
# LOAD CORD
# ============================================================

print("\nLoading CORD validation dataset...")

dataset = load_dataset("naver-clova-ix/cord-v2")

validation_images = dataset["validation"]

print("CORD validation images:", len(validation_images))


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

base_model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("Model loaded.")
print("Model device:", next(model.parameters()).device)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Extract the following information from this receipt/invoice.

Return ONLY a valid JSON object.

Use this exact structure:

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
- Extract values only if visible.
- If a value is not visible, use null.
- Do not add explanations.
- Do not use markdown.
- Keep numbers as strings.
"""


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower().strip()

    value = value.replace(",", "")
    value = value.replace("₹", "")
    value = value.replace("$", "")
    value = value.replace("rs.", "")
    value = value.replace("rs", "")

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_number(value):
    if value is None:
        return ""

    value = str(value).lower().strip()

    value = value.replace(",", "")
    value = value.replace("₹", "")
    value = value.replace("$", "")
    value = value.replace("rs.", "")
    value = value.replace("rs", "")

    value = re.sub(r"\s+", "", value)

    return value


# ============================================================
# JSON PARSER
# ============================================================

def extract_json(text):

    text = text.strip()

    # Direct JSON
    try:
        return json.loads(text)
    except:
        pass

    # Find JSON object inside output
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except:
            pass

    return None


# ============================================================
# METRICS
# ============================================================

def evaluate_prediction(gt, pred):

    result = {
        "json_valid": pred is not None,
        "total_correct": False,
        "name_correct": 0,
        "quantity_correct": 0,
        "price_correct": 0,
        "complete_correct": 0,
        "gt_items": 0,
        "pred_items": 0,
        "matched_items": 0
    }

    if pred is None:
        return result

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    gt_total = normalize_number(gt.get("total_amount"))
    pred_total = normalize_number(pred.get("total_amount"))

    if gt_total and pred_total and gt_total == pred_total:
        result["total_correct"] = True

    # --------------------------------------------------------
    # LINE ITEMS
    # --------------------------------------------------------

    gt_items = gt.get("line_items", [])
    pred_items = pred.get("line_items", [])

    result["gt_items"] = len(gt_items)
    result["pred_items"] = len(pred_items)

    used_predictions = set()

    for gt_item in gt_items:

        gt_name = normalize_text(gt_item.get("name"))
        gt_qty = normalize_number(gt_item.get("quantity"))
        gt_price = normalize_number(gt_item.get("price"))

        best_index = None

        # Exact name matching
        for i, pred_item in enumerate(pred_items):

            if i in used_predictions:
                continue

            pred_name = normalize_text(pred_item.get("name"))

            if gt_name and pred_name and gt_name == pred_name:
                best_index = i
                break

        if best_index is None:
            continue

        used_predictions.add(best_index)

        result["matched_items"] += 1

        pred_item = pred_items[best_index]

        pred_qty = normalize_number(pred_item.get("quantity"))
        pred_price = normalize_number(pred_item.get("price"))

        name_ok = gt_name == normalize_text(pred_item.get("name"))
        qty_ok = gt_qty == pred_qty
        price_ok = gt_price == pred_price

        if name_ok:
            result["name_correct"] += 1

        if qty_ok:
            result["quantity_correct"] += 1

        if price_ok:
            result["price_correct"] += 1

        if name_ok and qty_ok and price_ok:
            result["complete_correct"] += 1

    return result


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================

def run_configuration(config_name, max_width, max_height, max_tokens):

    print("\n")
    print("=" * 70)
    print("CONFIGURATION:", config_name)
    print("Image:", f"{max_width}x{max_height}")
    print("Max tokens:", max_tokens)
    print("=" * 70)

    times = []

    json_valid = 0
    total_correct = 0

    name_correct = 0
    quantity_correct = 0
    price_correct = 0
    complete_correct = 0

    gt_items = 0
    pred_items = 0
    matched_items = 0

    peak_memory = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for index in range(NUM_SAMPLES):

        print(
            f"\nSample {index + 1}/{NUM_SAMPLES}",
            end=" "
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt_record = ground_truth_records[index]
        gt = gt_record["target"]

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image = validation_images[index]["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        # Aspect preserving resize
        image.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS
        )

        # ----------------------------------------------------
        # Messages
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

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
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True
            )

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        times.append(elapsed)

        # ----------------------------------------------------
        # Decode only generated tokens
        # ----------------------------------------------------

        input_length = inputs["input_ids"].shape[1]

        generated_only = generated_ids[:, input_length:]

        output_text = processor.batch_decode(
            generated_only,
            skip_special_tokens=True
        )[0]

        prediction = extract_json(output_text)

        metrics = evaluate_prediction(
            gt,
            prediction
        )

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        if metrics["json_valid"]:
            json_valid += 1

        if metrics["total_correct"]:
            total_correct += 1

        name_correct += metrics["name_correct"]
        quantity_correct += metrics["quantity_correct"]
        price_correct += metrics["price_correct"]
        complete_correct += metrics["complete_correct"]

        gt_items += metrics["gt_items"]
        pred_items += metrics["pred_items"]
        matched_items += metrics["matched_items"]

        current_peak = torch.cuda.max_memory_allocated() / (
            1024 ** 3
        )

        peak_memory = max(
            peak_memory,
            current_peak
        )

        status = "YES" if metrics["json_valid"] else "NO"

        print(
            f"| Time: {elapsed:.2f}s | "
            f"JSON: {status}"
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    avg_time = sum(times) / len(times)

    json_accuracy = (
        json_valid / NUM_SAMPLES * 100
    )

    total_accuracy = (
        total_correct / NUM_SAMPLES * 100
    )

    if gt_items > 0:

        name_accuracy = (
            name_correct / gt_items * 100
        )

        quantity_accuracy = (
            quantity_correct / gt_items * 100
        )

        price_accuracy = (
            price_correct / gt_items * 100
        )

        complete_accuracy = (
            complete_correct / gt_items * 100
        )

        precision = (
            matched_items / pred_items * 100
            if pred_items > 0 else 0
        )

        recall = (
            matched_items / gt_items * 100
            if gt_items > 0 else 0
        )

        if precision + recall > 0:

            f1 = (
                2 * precision * recall
                / (precision + recall)
            )

        else:
            f1 = 0

    else:

        name_accuracy = 0
        quantity_accuracy = 0
        price_accuracy = 0
        complete_accuracy = 0
        precision = 0
        recall = 0
        f1 = 0

    print("\nRESULT:", config_name)

    print(
        f"JSON accuracy:       {json_accuracy:.2f}%"
    )

    print(
        f"Total accuracy:      {total_accuracy:.2f}%"
    )

    print(
        f"Name accuracy:       {name_accuracy:.2f}%"
    )

    print(
        f"Quantity accuracy:   {quantity_accuracy:.2f}%"
    )

    print(
        f"Price accuracy:      {price_accuracy:.2f}%"
    )

    print(
        f"Complete item:       {complete_accuracy:.2f}%"
    )

    print(
        f"Precision:           {precision:.2f}%"
    )

    print(
        f"Recall:              {recall:.2f}%"
    )

    print(
        f"F1 score:            {f1:.2f}%"
    )

    print(
        f"Average time:        {avg_time:.2f}s"
    )

    print(
        f"Fastest:             {min(times):.2f}s"
    )

    print(
        f"Slowest:             {max(times):.2f}s"
    )

    print(
        f"Peak VRAM:           {peak_memory:.2f} GB"
    )

    return {
        "config": config_name,
        "json_accuracy": json_accuracy,
        "total_accuracy": total_accuracy,
        "name_accuracy": name_accuracy,
        "quantity_accuracy": quantity_accuracy,
        "price_accuracy": price_accuracy,
        "complete_accuracy": complete_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_time": avg_time,
        "fastest": min(times),
        "slowest": max(times),
        "peak_vram": peak_memory
    }


# ============================================================
# RUN ALL CONFIGURATIONS
# ============================================================

results = []

for config in CONFIGS:

    result = run_configuration(*config)

    results.append(result)

    torch.cuda.empty_cache()


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n")
print("=" * 110)
print("FINAL CONFIGURATION COMPARISON")
print("=" * 110)

print(
    f"{'Config':<18}"
    f"{'JSON':>9}"
    f"{'Total':>9}"
    f"{'Name':>9}"
    f"{'Qty':>9}"
    f"{'Price':>9}"
    f"{'Complete':>11}"
    f"{'F1':>9}"
    f"{'Time':>10}"
    f"{'VRAM':>9}"
)

print("-" * 110)

for r in results:

    print(
        f"{r['config']:<18}"
        f"{r['json_accuracy']:>8.1f}%"
        f"{r['total_accuracy']:>8.1f}%"
        f"{r['name_accuracy']:>8.1f}%"
        f"{r['quantity_accuracy']:>8.1f}%"
        f"{r['price_accuracy']:>8.1f}%"
        f"{r['complete_accuracy']:>10.1f}%"
        f"{r['f1']:>8.1f}%"
        f"{r['avg_time']:>9.2f}s"
        f"{r['peak_vram']:>8.2f}G"
    )


# ============================================================
# SELECT BEST CONFIGURATION
# ============================================================

# Primary: complete item accuracy
# Secondary: F1
# Tertiary: inference speed

best = sorted(
    results,
    key=lambda r: (
        r["complete_accuracy"],
        r["f1"],
        -r["avg_time"]
    ),
    reverse=True
)[0]


print("\n")
print("=" * 70)
print("SELECTED CONFIGURATION")
print("=" * 70)

print("Configuration:", best["config"])
print("Complete item accuracy:", f"{best['complete_accuracy']:.2f}%")
print("F1 score:", f"{best['f1']:.2f}%")
print("Average inference:", f"{best['avg_time']:.2f}s")
print("Peak VRAM:", f"{best['peak_vram']:.2f} GB")

print("\nPHASE 8.10C COMPLETE")
print("=" * 70)