import os
import re
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
# CONFIGURATION
# ============================================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "outputs/qlora_mini"

VALIDATION_FILE = "data/processed/validation.jsonl"
OUTPUT_FILE = "outputs/final_validation_results.json"

NUM_SAMPLES = 20

MAX_IMAGE_SIZE = 384
MAX_NEW_TOKENS = 256


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text):

    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences
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

    # Try direct JSON parsing
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

        return None

    except Exception:
        pass

    # Try extracting JSON object
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

    return value


def normalize_number(value):

    if value is None:
        return ""

    value = str(value)

    value = value.replace(",", "")
    value = value.replace("₹", "")
    value = value.replace("$", "")
    value = value.replace("rs.", "")
    value = value.replace("rs", "")

    value = value.strip()

    # Remove unnecessary decimal zeros
    try:

        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(number)

    except Exception:

        return normalize_text(value)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation_data():

    samples = []

    with open(
        VALIDATION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                samples.append(
                    json.loads(line)
                )

    return samples[:NUM_SAMPLES]


# ============================================================
# IMAGE RESIZE
# ============================================================

def resize_image(image):

    image = image.convert("RGB")

    width, height = image.size

    longest_side = max(
        width,
        height
    )

    if longest_side > MAX_IMAGE_SIZE:

        scale = MAX_IMAGE_SIZE / longest_side

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize(
            (new_width, new_height)
        )

    return image


# ============================================================
# ITEM MATCHING
# ============================================================

def match_items(gt_items, pred_items):

    matched = []

    used_predictions = set()

    for gt in gt_items:

        gt_name = normalize_text(
            gt.get("name")
        )

        best_index = None

        # Exact name match
        for i, pred in enumerate(pred_items):

            if i in used_predictions:
                continue

            pred_name = normalize_text(
                pred.get("name")
            )

            if gt_name and pred_name:

                if gt_name == pred_name:

                    best_index = i
                    break

        # Partial name match
        if best_index is None:

            for i, pred in enumerate(pred_items):

                if i in used_predictions:
                    continue

                pred_name = normalize_text(
                    pred.get("name")
                )

                if gt_name and pred_name:

                    if (
                        gt_name in pred_name
                        or pred_name in gt_name
                    ):

                        best_index = i
                        break

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
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("PHASE 10 - FINAL VALIDATION")
    print("=" * 75)

    print()
    print("Model:", MODEL_ID)
    print("Adapter:", ADAPTER_PATH)
    print("Image size:", MAX_IMAGE_SIZE)
    print("Max tokens:", MAX_NEW_TOKENS)
    print("Samples:", NUM_SAMPLES)

    # --------------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

    print()
    print("CUDA device:")
    print(torch.cuda.get_device_name(0))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # QUANTIZATION
    # --------------------------------------------------------

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # --------------------------------------------------------
    # LOAD PROCESSOR
    # --------------------------------------------------------

    print()
    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    # --------------------------------------------------------
    # LOAD BASE MODEL
    # --------------------------------------------------------

    print("Loading base model...")

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # --------------------------------------------------------
    # LOAD FINAL QLORA
    # --------------------------------------------------------

    print("Loading QLoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH
    )

    model.eval()

    print("Model ready.")

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    validation_samples = load_validation_data()

    print()
    print(
        "Validation samples loaded:",
        len(validation_samples)
    )

    # --------------------------------------------------------
    # LOAD RAW CORD IMAGES
    # --------------------------------------------------------

    print("Loading CORD validation images...")

    cord_dataset = load_dataset(
        "naver-clova-ix/cord-v2"
    )

    cord_validation = cord_dataset[
        "validation"
    ]

    print("CORD images ready.")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    json_correct = 0

    total_correct = 0

    total_gt_items = 0
    total_pred_items = 0
    total_matched_items = 0

    quantity_correct = 0
    quantity_total = 0

    price_correct = 0
    price_total = 0

    complete_correct = 0
    complete_total = 0

    inference_times = []

    sample_results = []

    # --------------------------------------------------------
    # EVALUATION LOOP
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("RUNNING FINAL VALIDATION")
    print("=" * 75)

    for index, sample in enumerate(
        validation_samples
    ):

        image_id = sample["image_id"]

        target = sample["target"]

        gt_items = target.get(
            "line_items",
            []
        )

        gt_total = normalize_number(
            target.get("total_amount")
        )

        image = cord_validation[
            image_id
        ]["image"]

        image = resize_image(
            image
        )

        prompt = """
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
                        "text": prompt
                    }
                ]
            }
        ]

        # ----------------------------------------------------
        # PREPARE INPUT
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
            k: v.to(device)
            if hasattr(v, "to")
            else v
            for k, v in inputs.items()
        }

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True
            )

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        inference_times.append(
            elapsed
        )

        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        input_length = inputs[
            "input_ids"
        ].shape[1]

        generated_tokens = generated_ids[
            :, input_length:
        ]

        output_text = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0].strip()

        prediction = parse_json(
            output_text
        )

        # ----------------------------------------------------
        # JSON METRIC
        # ----------------------------------------------------

        json_valid = (
            prediction is not None
        )

        if json_valid:

            json_correct += 1

        # ----------------------------------------------------
        # DEFAULT PREDICTION
        # ----------------------------------------------------

        if prediction is None:

            prediction = {
                "line_items": []
            }

        pred_items = prediction.get(
            "line_items",
            []
        )

        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        pred_total = normalize_number(
            prediction.get("total_amount")
        )

        if (
            gt_total
            and pred_total
            and gt_total == pred_total
        ):

            total_correct += 1

        # ----------------------------------------------------
        # ITEM COUNTS
        # ----------------------------------------------------

        total_gt_items += len(
            gt_items
        )

        total_pred_items += len(
            pred_items
        )

        # ----------------------------------------------------
        # MATCH ITEMS
        # ----------------------------------------------------

        matched = match_items(
            gt_items,
            pred_items
        )

        total_matched_items += len(
            matched
        )

        # ----------------------------------------------------
        # FIELD ACCURACY
        # ----------------------------------------------------

        sample_quantity_correct = 0
        sample_price_correct = 0
        sample_complete = 0

        for gt, pred in matched:

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

            if gt_quantity:

                quantity_total += 1

                if (
                    gt_quantity
                    == pred_quantity
                ):

                    quantity_correct += 1
                    sample_quantity_correct += 1

            if gt_price:

                price_total += 1

                if (
                    gt_price
                    == pred_price
                ):

                    price_correct += 1
                    sample_price_correct += 1

            if (
                gt_quantity
                and gt_price
                and gt_quantity == pred_quantity
                and gt_price == pred_price
            ):

                complete_correct += 1
                sample_complete += 1

        complete_total += len(
            matched
        )

        # ----------------------------------------------------
        # SAMPLE RESULT
        # ----------------------------------------------------

        sample_results.append(
            {
                "sample": index + 1,
                "image_id": image_id,
                "json_valid": json_valid,
                "ground_truth_items": len(gt_items),
                "predicted_items": len(pred_items),
                "matched_items": len(matched),
                "time_seconds": round(
                    elapsed,
                    3
                ),
                "quantity_correct": sample_quantity_correct,
                "price_correct": sample_price_correct,
                "complete_correct": sample_complete
            }
        )

        print(
            f"Sample {index + 1:02d}/{len(validation_samples)} "
            f"| {elapsed:.2f}s "
            f"| GT={len(gt_items)} "
            f"| Pred={len(pred_items)} "
            f"| Matched={len(matched)} "
            f"| JSON={'OK' if json_valid else 'FAIL'}"
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    json_accuracy = (
        json_correct / len(validation_samples) * 100
        if validation_samples
        else 0
    )

    total_accuracy = (
        total_correct / len(validation_samples) * 100
        if validation_samples
        else 0
    )

    quantity_accuracy = (
        quantity_correct / quantity_total * 100
        if quantity_total
        else 0
    )

    price_accuracy = (
        price_correct / price_total * 100
        if price_total
        else 0
    )

    complete_accuracy = (
        complete_correct / complete_total * 100
        if complete_total
        else 0
    )

    precision = (
        total_matched_items
        / total_pred_items
        * 100
        if total_pred_items
        else 0
    )

    recall = (
        total_matched_items
        / total_gt_items
        * 100
        if total_gt_items
        else 0
    )

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0

    average_time = (
        sum(inference_times)
        / len(inference_times)
        if inference_times
        else 0
    )

    fastest_time = (
        min(inference_times)
        if inference_times
        else 0
    )

    slowest_time = (
        max(inference_times)
        if inference_times
        else 0
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    final_allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    final_reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 75)
    print("PHASE 10 - FINAL VALIDATION RESULTS")
    print("=" * 75)

    print(
        f"Samples tested:          {len(validation_samples)}"
    )

    print(
        f"JSON accuracy:           {json_accuracy:.2f}%"
    )

    print(
        f"Total accuracy:          {total_accuracy:.2f}%"
    )

    print(
        f"Quantity accuracy:       {quantity_accuracy:.2f}%"
    )

    print(
        f"Price accuracy:          {price_accuracy:.2f}%"
    )

    print(
        f"Complete item accuracy:  {complete_accuracy:.2f}%"
    )

    print(
        f"Precision:               {precision:.2f}%"
    )

    print(
        f"Recall:                  {recall:.2f}%"
    )

    print(
        f"F1 score:                {f1:.2f}%"
    )

    print()
    print(
        f"Average inference:      {average_time:.2f} sec"
    )

    print(
        f"Fastest inference:      {fastest_time:.2f} sec"
    )

    print(
        f"Slowest inference:      {slowest_time:.2f} sec"
    )

    print(
        f"Peak GPU memory:        {peak_memory:.2f} GB"
    )

    print(
        f"Final GPU allocated:    {final_allocated:.2f} GB"
    )

    print(
        f"Final GPU reserved:     {final_reserved:.2f} GB"
    )

    print()
    print(
        f"GT items:                {total_gt_items}"
    )

    print(
        f"Predicted items:         {total_pred_items}"
    )

    print(
        f"Matched items:           {total_matched_items}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {

        "phase": "10 - Final Validation",

        "model": MODEL_ID,

        "adapter": ADAPTER_PATH,

        "configuration": {
            "quantization": "4-bit NF4",
            "compute_dtype": "bfloat16",
            "image_max_size": MAX_IMAGE_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "use_cache": True,
            "inference_mode": True
        },

        "samples_tested": len(
            validation_samples
        ),

        "metrics": {

            "json_accuracy": round(
                json_accuracy,
                2
            ),

            "total_accuracy": round(
                total_accuracy,
                2
            ),

            "quantity_accuracy": round(
                quantity_accuracy,
                2
            ),

            "price_accuracy": round(
                price_accuracy,
                2
            ),

            "complete_item_accuracy": round(
                complete_accuracy,
                2
            ),

            "precision": round(
                precision,
                2
            ),

            "recall": round(
                recall,
                2
            ),

            "f1": round(
                f1,
                2
            )
        },

        "performance": {

            "average_inference_seconds": round(
                average_time,
                3
            ),

            "fastest_seconds": round(
                fastest_time,
                3
            ),

            "slowest_seconds": round(
                slowest_time,
                3
            ),

            "peak_gpu_memory_gb": round(
                peak_memory,
                3
            ),

            "final_gpu_allocated_gb": round(
                final_allocated,
                3
            ),

            "final_gpu_reserved_gb": round(
                final_reserved,
                3
            )
        },

        "items": {

            "ground_truth": total_gt_items,

            "predicted": total_pred_items,

            "matched": total_matched_items
        },

        "samples": sample_results
    }

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
            results,
            f,
            indent=2
        )

    print()
    print(
        "Final results saved to:",
        OUTPUT_FILE
    )

    print()
    print("=" * 75)
    print("PHASE 10 COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()