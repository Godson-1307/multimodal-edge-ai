import json
import re
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
ADAPTER_DIR = "outputs/qlora_mini"

NUM_SAMPLES = 20
MAX_IMAGE_SIZE = 768


# ============================================================
# LOAD DATASET
# ============================================================

dataset = load_dataset(
    "naver-clova-ix/cord-v2",
    split="validation"
)

eval_dataset = dataset.select(range(NUM_SAMPLES))


print("\n============================================")
print("PHASE 7.2 - EXTRACTION METRICS")
print("============================================")
print("Validation samples:", len(dataset))
print("Evaluation samples:", NUM_SAMPLES)


# ============================================================
# CORD TEXT EXTRACTION
# ============================================================

def get_text(annotation):

    words = annotation.get("words", [])

    text = " ".join(
        word.get("text", "").strip()
        for word in words
        if word.get("text", "").strip()
    )

    return text.strip()


def find_annotations(data):

    if isinstance(data, list):

        if data and all(
            isinstance(item, dict) and "category" in item
            for item in data
        ):
            return data

        for item in data:

            result = find_annotations(item)

            if result is not None:
                return result

    elif isinstance(data, dict):

        for value in data.values():

            result = find_annotations(value)

            if result is not None:
                return result

    return None


def extract_ground_truth(ground_truth_json):

    ground_truth = json.loads(ground_truth_json)

    annotations = find_annotations(ground_truth)

    if annotations is None:
        raise ValueError(
            "Could not find CORD annotation list."
        )

    target = {
        "total_tax": None,
        "total_amount": None,
        "line_items": []
    }

    menu_items = {}

    for annotation in annotations:

        category = annotation.get("category")

        group_id = annotation.get("group_id")

        text = get_text(annotation)


        # -----------------------------
        # LINE ITEMS
        # -----------------------------

        if category and category.startswith("menu."):

            if group_id not in menu_items:

                menu_items[group_id] = {
                    "name": None,
                    "quantity": None,
                    "price": None
                }


            if category == "menu.nm":

                menu_items[group_id]["name"] = text


            elif category == "menu.cnt":

                parts = text.split()

                if parts:

                    menu_items[group_id]["quantity"] = parts[0]


            elif category == "menu.price":

                menu_items[group_id]["price"] = text


        # -----------------------------
        # TAX
        # -----------------------------

        elif category == "sub_total.tax_price":

            target["total_tax"] = text


        # -----------------------------
        # TOTAL
        # -----------------------------

        elif category == "total.total_price":

            target["total_amount"] = text


    # -----------------------------
    # SORT LINE ITEMS
    # -----------------------------

    for group_id in sorted(menu_items):

        item = menu_items[group_id]

        target["line_items"].append({
            "name": item["name"],
            "quantity": item["quantity"],
            "price": item["price"]
        })


    return target


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def normalize_text(value):

    if value is None:
        return None

    value = str(value)

    value = value.lower()

    value = value.strip()

    value = re.sub(r"\s+", " ", value)

    return value


def normalize_amount(value):

    if value is None:
        return None

    value = str(value)

    # Remove everything except digits
    digits = re.sub(r"[^0-9]", "", value)

    if digits == "":
        return None

    return digits


def normalize_quantity(value):

    if value is None:
        return None

    value = str(value)

    # Keep digits only
    digits = re.sub(r"[^0-9]", "", value)

    if digits == "":
        return None

    return digits


# ============================================================
# PARSE MODEL OUTPUT
# ============================================================

def parse_prediction(prediction_text):

    prediction_text = prediction_text.strip()


    # Remove markdown JSON block

    if prediction_text.startswith("```json"):

        prediction_text = prediction_text[7:]


    elif prediction_text.startswith("```"):

        prediction_text = prediction_text[3:]


    if prediction_text.endswith("```"):

        prediction_text = prediction_text[:-3]


    prediction_text = prediction_text.strip()


    try:

        prediction = json.loads(prediction_text)

        return prediction, True

    except json.JSONDecodeError:

        return None, False


# ============================================================
# IMAGE RESIZE
# ============================================================

def resize_image(image):

    width, height = image.size

    scale = min(
        MAX_IMAGE_SIZE / width,
        MAX_IMAGE_SIZE / height,
        1
    )

    new_width = int(width * scale)

    new_height = int(height * scale)


    if scale < 1:

        image = image.resize(
            (new_width, new_height)
        )


    return image


# ============================================================
# GENERATE PREDICTION
# ============================================================

def generate_prediction(model, processor, image):

    image = resize_image(image)


    prompt = """
Extract the following information from this receipt:

- total_tax
- total_amount
- line_items

For each line item extract:
- name
- quantity
- price

Return ONLY a valid JSON object.

Do not add explanations.
Do not use markdown code blocks.

If a field is not visible, use null.
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
        if hasattr(value, "to")
        else value

        for key, value in inputs.items()
    }


    start_time = time.time()


    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512
        )


    inference_time = time.time() - start_time


    generated_ids = (
        output_ids[:, inputs["input_ids"].shape[1]:]
    )


    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()


    return output_text, inference_time


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========== LOADING MODEL ==========")


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


base_model.eval()


print("Base model loaded successfully.")


# ============================================================
# RUN BASE MODEL
# ============================================================

print("\n============================================")
print("BASE MODEL EVALUATION")
print("============================================")


base_results = []


for i, sample in enumerate(eval_dataset):

    print(
        f"\n----- BASE SAMPLE {i + 1} -----"
    )


    ground_truth = extract_ground_truth(
        sample["ground_truth"]
    )


    prediction_text, inference_time = (
        generate_prediction(
            base_model,
            processor,
            sample["image"]
        )
    )


    prediction, json_valid = parse_prediction(
        prediction_text
    )


    if not json_valid:

        prediction = {
            "total_tax": None,
            "total_amount": None,
            "line_items": []
        }


    base_results.append({

        "ground_truth": ground_truth,

        "prediction": prediction,

        "json_valid": json_valid,

        "inference_time": inference_time

    })


    print(
        "JSON Valid:",
        json_valid
    )


    print(
        "Inference Time:",
        round(inference_time, 2),
        "seconds"
    )


# ============================================================
# LOAD LORA ADAPTER
# ============================================================

print("\n============================================")
print("LOADING QLORA ADAPTER")
print("============================================")


finetuned_model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_DIR
)


finetuned_model.eval()


print("QLoRA adapter loaded successfully.")


# ============================================================
# RUN FINETUNED MODEL
# ============================================================

print("\n============================================")
print("FINE-TUNED MODEL EVALUATION")
print("============================================")


finetuned_results = []


for i, sample in enumerate(eval_dataset):

    print(
        f"\n----- FINE-TUNED SAMPLE {i + 1} -----"
    )


    ground_truth = extract_ground_truth(
        sample["ground_truth"]
    )


    prediction_text, inference_time = (
        generate_prediction(
            finetuned_model,
            processor,
            sample["image"]
        )
    )


    prediction, json_valid = parse_prediction(
        prediction_text
    )


    if not json_valid:

        prediction = {
            "total_tax": None,
            "total_amount": None,
            "line_items": []
        }


    finetuned_results.append({

        "ground_truth": ground_truth,

        "prediction": prediction,

        "json_valid": json_valid,

        "inference_time": inference_time

    })


    print(
        "JSON Valid:",
        json_valid
    )


    print(
        "Inference Time:",
        round(inference_time, 2),
        "seconds"
    )


# ============================================================
# LINE ITEM MATCHING
# ============================================================

def match_line_items(
    ground_truth_items,
    predicted_items
):

    matched = []

    used_predictions = set()


    for gt_item in ground_truth_items:

        gt_name = normalize_text(
            gt_item.get("name")
        )


        best_index = None


        for index, pred_item in enumerate(
            predicted_items
        ):

            if index in used_predictions:
                continue


            pred_name = normalize_text(
                pred_item.get("name")
            )


            if (
                gt_name is not None
                and pred_name == gt_name
            ):

                best_index = index

                break


        if best_index is not None:

            used_predictions.add(best_index)

            matched.append(
                (
                    gt_item,
                    predicted_items[best_index]
                )
            )


    return matched


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(results):

    json_valid = 0

    total_amount_correct = 0

    total_amount_available = 0


    total_gt_items = 0

    total_pred_items = 0

    matched_items = 0

    name_correct = 0

    quantity_correct = 0

    price_correct = 0

    complete_items_correct = 0


    total_inference_time = 0


    for result in results:

        gt = result["ground_truth"]

        pred = result["prediction"]


        # -----------------------------
        # JSON VALIDITY
        # -----------------------------

        if result["json_valid"]:

            json_valid += 1


        # -----------------------------
        # TOTAL AMOUNT
        # -----------------------------

        gt_total = normalize_amount(
            gt.get("total_amount")
        )

        pred_total = normalize_amount(
            pred.get("total_amount")
        )


        if gt_total is not None:

            total_amount_available += 1


            if pred_total == gt_total:

                total_amount_correct += 1


        # -----------------------------
        # LINE ITEMS
        # -----------------------------

        gt_items = gt.get(
            "line_items",
            []
        )


        pred_items = pred.get(
            "line_items",
            []
        )


        total_gt_items += len(gt_items)

        total_pred_items += len(pred_items)


        matches = match_line_items(
            gt_items,
            pred_items
        )


        matched_items += len(matches)


        for gt_item, pred_item in matches:

            # NAME

            gt_name = normalize_text(
                gt_item.get("name")
            )

            pred_name = normalize_text(
                pred_item.get("name")
            )


            if gt_name == pred_name:

                name_correct += 1


            # QUANTITY

            gt_quantity = normalize_quantity(
                gt_item.get("quantity")
            )

            pred_quantity = normalize_quantity(
                pred_item.get("quantity")
            )


            if gt_quantity == pred_quantity:

                quantity_correct += 1


            # PRICE

            gt_price = normalize_amount(
                gt_item.get("price")
            )

            pred_price = normalize_amount(
                pred_item.get("price")
            )


            if gt_price == pred_price:

                price_correct += 1


            # COMPLETE ITEM

            if (
                gt_name == pred_name
                and gt_quantity == pred_quantity
                and gt_price == pred_price
            ):

                complete_items_correct += 1


        total_inference_time += (
            result["inference_time"]
        )


    # ========================================================
    # FINAL PERCENTAGES
    # ========================================================

    json_validity_rate = (
        json_valid / NUM_SAMPLES
    ) * 100


    if total_amount_available > 0:

        total_amount_accuracy = (
            total_amount_correct
            / total_amount_available
        ) * 100

    else:

        total_amount_accuracy = 0


    if total_gt_items > 0:

        name_accuracy = (
            name_correct
            / total_gt_items
        ) * 100

        quantity_accuracy = (
            quantity_correct
            / total_gt_items
        ) * 100

        price_accuracy = (
            price_correct
            / total_gt_items
        ) * 100

        complete_item_accuracy = (
            complete_items_correct
            / total_gt_items
        ) * 100

    else:

        name_accuracy = 0

        quantity_accuracy = 0

        price_accuracy = 0

        complete_item_accuracy = 0


    return {

        "json_validity": json_validity_rate,

        "total_amount_accuracy":
            total_amount_accuracy,

        "name_accuracy":
            name_accuracy,

        "quantity_accuracy":
            quantity_accuracy,

        "price_accuracy":
            price_accuracy,

        "complete_item_accuracy":
            complete_item_accuracy,

        "ground_truth_items":
            total_gt_items,

        "predicted_items":
            total_pred_items,

        "matched_items":
            matched_items,

        "average_inference_time":
            total_inference_time / NUM_SAMPLES
    }


# ============================================================
# CALCULATE BASE METRICS
# ============================================================

base_metrics = calculate_metrics(
    base_results
)


# ============================================================
# CALCULATE FINE-TUNED METRICS
# ============================================================

finetuned_metrics = calculate_metrics(
    finetuned_results
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n\n============================================")
print("FINAL METRICS")
print("============================================")


print("\nBASE MODEL")
print("--------------------------------------------")

print(
    "JSON Validity:",
    f"{base_metrics['json_validity']:.2f}%"
)

print(
    "Total Amount Accuracy:",
    f"{base_metrics['total_amount_accuracy']:.2f}%"
)

print(
    "Line Item Name Accuracy:",
    f"{base_metrics['name_accuracy']:.2f}%"
)

print(
    "Quantity Accuracy:",
    f"{base_metrics['quantity_accuracy']:.2f}%"
)

print(
    "Price Accuracy:",
    f"{base_metrics['price_accuracy']:.2f}%"
)

print(
    "Complete Line Item Accuracy:",
    f"{base_metrics['complete_item_accuracy']:.2f}%"
)

print(
    "Ground Truth Items:",
    base_metrics["ground_truth_items"]
)

print(
    "Predicted Items:",
    base_metrics["predicted_items"]
)

print(
    "Matched Items:",
    base_metrics["matched_items"]
)

print(
    "Average Inference Time:",
    f"{base_metrics['average_inference_time']:.2f} seconds"
)


print("\nFINE-TUNED MODEL")
print("--------------------------------------------")

print(
    "JSON Validity:",
    f"{finetuned_metrics['json_validity']:.2f}%"
)

print(
    "Total Amount Accuracy:",
    f"{finetuned_metrics['total_amount_accuracy']:.2f}%"
)

print(
    "Line Item Name Accuracy:",
    f"{finetuned_metrics['name_accuracy']:.2f}%"
)

print(
    "Quantity Accuracy:",
    f"{finetuned_metrics['quantity_accuracy']:.2f}%"
)

print(
    "Price Accuracy:",
    f"{finetuned_metrics['price_accuracy']:.2f}%"
)

print(
    "Complete Line Item Accuracy:",
    f"{finetuned_metrics['complete_item_accuracy']:.2f}%"
)

print(
    "Ground Truth Items:",
    finetuned_metrics["ground_truth_items"]
)

print(
    "Predicted Items:",
    finetuned_metrics["predicted_items"]
)

print(
    "Matched Items:",
    finetuned_metrics["matched_items"]
)

print(
    "Average Inference Time:",
    f"{finetuned_metrics['average_inference_time']:.2f} seconds"
)


# ============================================================
# IMPROVEMENT
# ============================================================

print("\n============================================")
print("BASE vs FINE-TUNED IMPROVEMENT")
print("============================================")


print(
    "Total Amount:",
    f"{base_metrics['total_amount_accuracy']:.2f}% -> "
    f"{finetuned_metrics['total_amount_accuracy']:.2f}%"
)


print(
    "Name Accuracy:",
    f"{base_metrics['name_accuracy']:.2f}% -> "
    f"{finetuned_metrics['name_accuracy']:.2f}%"
)


print(
    "Quantity Accuracy:",
    f"{base_metrics['quantity_accuracy']:.2f}% -> "
    f"{finetuned_metrics['quantity_accuracy']:.2f}%"
)


print(
    "Price Accuracy:",
    f"{base_metrics['price_accuracy']:.2f}% -> "
    f"{finetuned_metrics['price_accuracy']:.2f}%"
)


print(
    "Complete Item Accuracy:",
    f"{base_metrics['complete_item_accuracy']:.2f}% -> "
    f"{finetuned_metrics['complete_item_accuracy']:.2f}%"
)


print(
    "Average Inference:",
    f"{base_metrics['average_inference_time']:.2f}s -> "
    f"{finetuned_metrics['average_inference_time']:.2f}s"
)


print("\n============================================")
print("STEP 7.2 SUCCESS")
print("Metrics evaluation completed.")
print("============================================")