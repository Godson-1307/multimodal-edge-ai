import time
import json
import re

import torch

from datasets import load_dataset

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


# ============================================
# CONFIGURATION
# ============================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "outputs/qlora_mini"

NUM_SAMPLES = 5

RESOLUTIONS = [
    (384, 576),
    (512, 768),
    (640, 960),
    (768, 1152)
]


# ============================================
# GPU MEMORY FUNCTION
# ============================================

def gpu_memory():

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    return allocated, reserved


# ============================================
# TEXT NORMALIZATION
# ============================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value).lower().strip()

    value = re.sub(r"\s+", " ", value)

    return value


# ============================================
# AMOUNT NORMALIZATION
# ============================================

def normalize_amount(value):

    if value is None:
        return ""

    value = str(value)

    # Keep only digits
    digits = re.sub(r"[^\d]", "", value)

    return digits


# ============================================
# JSON PARSER
# ============================================

def parse_json_response(text):

    try:

        # Find JSON object inside model response
        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if not match:
            return None

        json_text = match.group()

        return json.loads(json_text)

    except Exception:

        return None


# ============================================
# FIND CORD ANNOTATIONS
# ============================================

def find_annotations(obj):

    annotations = []

    if isinstance(obj, dict):

        # Direct annotation list
        if "annotations" in obj:

            if isinstance(obj["annotations"], list):

                annotations.extend(
                    obj["annotations"]
                )

        # Search recursively
        for value in obj.values():

            annotations.extend(
                find_annotations(value)
            )

    elif isinstance(obj, list):

        for item in obj:

            annotations.extend(
                find_annotations(item)
            )

    return annotations


# ============================================
# GET WORD TEXT FROM CORD ANNOTATION
# ============================================

def get_annotation_text(annotation):

    words = annotation.get("words", [])

    if isinstance(words, list):

        text_parts = []

        for word in words:

            if isinstance(word, dict):

                text = word.get("text")

                if text:
                    text_parts.append(str(text))

            elif isinstance(word, str):

                text_parts.append(word)

        return " ".join(text_parts).strip()

    return ""


# ============================================
# CONVERT CORD GROUND TRUTH
# ============================================

def convert_cord_ground_truth(sample):

    raw_gt = sample["ground_truth"]

    # ----------------------------------------
    # Parse JSON string
    # ----------------------------------------

    if isinstance(raw_gt, str):

        try:
            raw_gt = json.loads(raw_gt)

        except Exception:
            return {
                "total_amount": None,
                "line_items": []
            }

    # ----------------------------------------
    # Get parsed ground truth
    # ----------------------------------------

    gt_parse = raw_gt.get("gt_parse", {})

    # ----------------------------------------
    # TOTAL AMOUNT
    # ----------------------------------------

    total_amount = None

    total_data = gt_parse.get("total", {})

    if isinstance(total_data, dict):

        total_amount = total_data.get(
            "total_price"
        )

    # ----------------------------------------
    # LINE ITEMS
    # ----------------------------------------

    line_items = []

    menu = gt_parse.get("menu", [])

    # ----------------------------------------
    # CASE 1: Multiple menu items
    # ----------------------------------------

    if isinstance(menu, list):

        for item in menu:

            if not isinstance(item, dict):
                continue

            line_item = {
                "name": item.get("nm"),
                "quantity": item.get("cnt"),
                "price": item.get("price")
            }

            if (
                line_item["name"] is None
                and line_item["quantity"] is None
                and line_item["price"] is None
            ):
                continue

            line_items.append(line_item)

    # ----------------------------------------
    # CASE 2: Single menu item
    # ----------------------------------------

    elif isinstance(menu, dict):

        line_item = {
            "name": menu.get("nm"),
            "quantity": menu.get("cnt"),
            "price": menu.get("price")
        }

        if (
            line_item["name"] is not None
            or line_item["quantity"] is not None
            or line_item["price"] is not None
        ):

            line_items.append(line_item)

    return {
        "total_amount": total_amount,
        "line_items": line_items
    }


# ============================================
# COMPARE PREDICTION WITH GROUND TRUTH
# ============================================

def evaluate_prediction(prediction, ground_truth):

    metrics = {
        "json_valid": 0,

        "total_correct": 0,

        "name_correct": 0,
        "quantity_correct": 0,
        "price_correct": 0,

        "complete_items": 0,

        "gt_items": 0,
        "pred_items": 0,
        "matched_items": 0
    }

    # ========================================
    # JSON VALIDITY
    # ========================================

    if prediction is None:

        return metrics

    metrics["json_valid"] = 1

    # ========================================
    # TOTAL AMOUNT
    # ========================================

    gt_total = normalize_amount(
        ground_truth.get("total_amount")
    )

    pred_total = normalize_amount(
        prediction.get("total_amount")
    )

    if (
        gt_total != ""
        and pred_total != ""
        and gt_total == pred_total
    ):

        metrics["total_correct"] = 1

    # ========================================
    # LINE ITEMS
    # ========================================

    gt_items = ground_truth.get(
        "line_items",
        []
    )

    pred_items = prediction.get(
        "line_items",
        []
    )

    if not isinstance(gt_items, list):

        gt_items = []

    if not isinstance(pred_items, list):

        pred_items = []

    metrics["gt_items"] = len(gt_items)
    metrics["pred_items"] = len(pred_items)

    # ========================================
    # MATCH ITEMS BY NAME
    # ========================================

    pred_lookup = {}

    for item in pred_items:

        if not isinstance(item, dict):

            continue

        name = normalize_text(
            item.get("name")
        )

        if name != "":

            pred_lookup[name] = item

    for gt_item in gt_items:

        if not isinstance(gt_item, dict):

            continue

        gt_name = normalize_text(
            gt_item.get("name")
        )

        if gt_name == "":

            continue

        # ------------------------------------
        # Find prediction with same name
        # ------------------------------------

        pred_item = pred_lookup.get(
            gt_name
        )

        if pred_item is None:

            continue

        # Item matched
        metrics["matched_items"] += 1

        # Name correct
        metrics["name_correct"] += 1

        # ------------------------------------
        # Quantity
        # ------------------------------------

        gt_quantity = normalize_text(
            gt_item.get("quantity")
        )

        pred_quantity = normalize_text(
            pred_item.get("quantity")
        )

        if (
            gt_quantity != ""
            and pred_quantity != ""
            and gt_quantity == pred_quantity
        ):

            metrics["quantity_correct"] += 1

        # ------------------------------------
        # Price
        # ------------------------------------

        gt_price = normalize_amount(
            gt_item.get("price")
        )

        pred_price = normalize_amount(
            pred_item.get("price")
        )

        if (
            gt_price != ""
            and pred_price != ""
            and gt_price == pred_price
        ):

            metrics["price_correct"] += 1

        # ------------------------------------
        # COMPLETE ITEM
        # ------------------------------------

        quantity_correct = (
            gt_quantity != ""
            and pred_quantity != ""
            and gt_quantity == pred_quantity
        )

        price_correct = (
            gt_price != ""
            and pred_price != ""
            and gt_price == pred_price
        )

        if quantity_correct and price_correct:

            metrics["complete_items"] += 1

    return metrics


# ============================================
# START
# ============================================

print("\n============================================")
print("PHASE 8.2B - RESOLUTION VS ACCURACY")
print("============================================")


# ============================================
# GPU
# ============================================

print("\nGPU:")

print(
    torch.cuda.get_device_name(0)
)


# ============================================
# LOAD PROCESSOR
# ============================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)


# ============================================
# QUANTIZATION
# ============================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)


# ============================================
# LOAD MODEL
# ============================================

print("Loading model...")

model = AutoModelForMultimodalLM.from_pretrained(

    MODEL_ID,

    quantization_config=bnb_config,

    device_map="auto"
)


# ============================================
# LOAD QLORA ADAPTER
# ============================================

print("Loading QLoRA adapter...")

model = PeftModel.from_pretrained(

    model,

    ADAPTER_PATH
)

model.eval()

print("Model ready.")


# ============================================
# LOAD DATASET
# ============================================

print("\nLoading validation dataset...")

dataset = load_dataset(

    "naver-clova-ix/cord-v2",

    split="validation"
)

dataset = dataset.select(
    range(NUM_SAMPLES)
)

print(
    "Samples:",
    len(dataset)
)


# ============================================
# PROMPT
# ============================================

prompt = """
Extract the following information from this invoice:

- vendor_name
- invoice_number
- invoice_date
- gstin
- state
- state_code
- total_tax
- total_amount
- line_items

For line_items extract:
- name
- quantity
- price

Return ONLY a valid JSON object.
Do not add explanations.
Do not use markdown code blocks.
If a field is not visible, use null.
"""


# ============================================
# RESULTS
# ============================================

results = []


# ============================================
# TEST EACH RESOLUTION
# ============================================

for max_width, max_height in RESOLUTIONS:

    print("\n============================================")

    print(
        "TESTING RESOLUTION:",
        max_width,
        "x",
        max_height
    )

    print("============================================")


    # ----------------------------------------
    # Timing
    # ----------------------------------------

    times = []


    # ----------------------------------------
    # Accuracy counters
    # ----------------------------------------

    total_json_valid = 0

    total_correct = 0

    name_correct = 0
    quantity_correct = 0
    price_correct = 0
    complete_items = 0

    gt_items = 0
    pred_items = 0
    matched_items = 0


    # ----------------------------------------
    # Reset GPU memory statistics
    # ----------------------------------------

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()


    # ========================================
    # PROCESS SAMPLES
    # ========================================

    for i, sample in enumerate(dataset):

        print(
            f"\n--- Sample {i + 1}/{NUM_SAMPLES} ---"
        )


        # ------------------------------------
        # IMAGE
        # ------------------------------------

        image = sample["image"].copy()

        original_size = image.size

        # Keep aspect ratio
        image.thumbnail(
            (
                max_width,
                max_height
            )
        )


        # ====================================
        # GROUND TRUTH
        # ====================================

        ground_truth = (
            convert_cord_ground_truth(
                sample
            )
        )


        # ====================================
        # CHAT MESSAGE
        # ====================================

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


        # ====================================
        # CHAT TEMPLATE
        # ====================================

        text = processor.apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True
        )


        # ====================================
        # PROCESS INPUT
        # ====================================

        inputs = processor(

            text=[text],

            images=[image],

            padding=True,

            return_tensors="pt"
        )


        # ====================================
        # MOVE INPUT TO GPU
        # ====================================

        inputs = {

            k: v.to(model.device)
            if torch.is_tensor(v)
            else v

            for k, v in inputs.items()

        }


        # ====================================
        # SYNCHRONIZE GPU
        # ====================================

        torch.cuda.synchronize()


        # ====================================
        # GENERATION
        # ====================================

        start = time.perf_counter()


        with torch.inference_mode():

            output_ids = model.generate(

                **inputs,

                max_new_tokens=256,

                do_sample=False

            )


        torch.cuda.synchronize()


        end = time.perf_counter()


        elapsed = end - start

        times.append(elapsed)


        # ====================================
        # DECODE ONLY GENERATED TOKENS
        # ====================================

        input_length = inputs["input_ids"].shape[1]

        generated_ids = output_ids[
            :,
            input_length:
        ]


        generated_text = processor.batch_decode(

            generated_ids,

            skip_special_tokens=True

        )[0].strip()


        # ====================================
        # PARSE JSON
        # ====================================

        prediction = parse_json_response(
            generated_text
        )


        # ====================================
        # EVALUATE
        # ====================================

        metrics = evaluate_prediction(

            prediction,

            ground_truth

        )


        # ====================================
        # ADD METRICS
        # ====================================

        total_json_valid += (
            metrics["json_valid"]
        )

        total_correct += (
            metrics["total_correct"]
        )

        name_correct += (
            metrics["name_correct"]
        )

        quantity_correct += (
            metrics["quantity_correct"]
        )

        price_correct += (
            metrics["price_correct"]
        )

        complete_items += (
            metrics["complete_items"]
        )

        gt_items += (
            metrics["gt_items"]
        )

        pred_items += (
            metrics["pred_items"]
        )

        matched_items += (
            metrics["matched_items"]
        )


        # ====================================
        # PRINT SAMPLE RESULT
        # ====================================

        print(
            "Image:",
            original_size,
            "->",
            image.size
        )

        print(
            "Time:",
            round(elapsed, 2),
            "sec"
        )

        print(
            "JSON:",
            "VALID"
            if prediction is not None
            else "INVALID"
        )

        print(
            "GT items:",
            metrics["gt_items"]
        )

        print(
            "Pred items:",
            metrics["pred_items"]
        )

        print(
            "Matched:",
            metrics["matched_items"]
        )


        # ====================================
        # CLEAN GPU MEMORY
        # ====================================

        del output_ids
        del generated_ids
        del inputs

        torch.cuda.empty_cache()


    # ========================================
    # CALCULATE RESULTS
    # ========================================

    average_time = (
        sum(times) / len(times)
    )


    fastest = min(times)

    slowest = max(times)


    peak_memory = (

        torch.cuda.max_memory_allocated()

        / (1024 ** 3)

    )


    # ========================================
    # ACCURACY PERCENTAGES
    # ========================================

    json_accuracy = (

        total_json_valid
        / NUM_SAMPLES
        * 100

    )


    total_accuracy = (

        total_correct
        / NUM_SAMPLES
        * 100

    )


    if gt_items > 0:

        name_accuracy = (

            name_correct
            / gt_items
            * 100

        )

        quantity_accuracy = (

            quantity_correct
            / gt_items
            * 100

        )

        price_accuracy = (

            price_correct
            / gt_items
            * 100

        )

        complete_accuracy = (

            complete_items
            / gt_items
            * 100

        )

    else:

        name_accuracy = 0
        quantity_accuracy = 0
        price_accuracy = 0
        complete_accuracy = 0


    # ========================================
    # SAVE RESULT
    # ========================================

    results.append({

        "resolution":
            f"{max_width}x{max_height}",

        "average_time":
            average_time,

        "fastest":
            fastest,

        "slowest":
            slowest,

        "peak_memory":
            peak_memory,

        "json_accuracy":
            json_accuracy,

        "total_accuracy":
            total_accuracy,

        "name_accuracy":
            name_accuracy,

        "quantity_accuracy":
            quantity_accuracy,

        "price_accuracy":
            price_accuracy,

        "complete_accuracy":
            complete_accuracy,

        "gt_items":
            gt_items,

        "pred_items":
            pred_items,

        "matched_items":
            matched_items

    })


    # ========================================
    # PRINT RESOLUTION RESULT
    # ========================================

    print("\n--------------------------------------------")

    print(
        "Resolution:",
        f"{max_width}x{max_height}"
    )

    print(
        "Average time:",
        round(average_time, 2),
        "sec"
    )

    print(
        "Fastest:",
        round(fastest, 2),
        "sec"
    )

    print(
        "Slowest:",
        round(slowest, 2),
        "sec"
    )

    print(
        "Peak VRAM:",
        round(peak_memory, 2),
        "GB"
    )

    print(
        "JSON accuracy:",
        round(json_accuracy, 2),
        "%"
    )

    print(
        "Total accuracy:",
        round(total_accuracy, 2),
        "%"
    )

    print(
        "Name accuracy:",
        round(name_accuracy, 2),
        "%"
    )

    print(
        "Quantity accuracy:",
        round(quantity_accuracy, 2),
        "%"
    )

    print(
        "Price accuracy:",
        round(price_accuracy, 2),
        "%"
    )

    print(
        "Complete item accuracy:",
        round(complete_accuracy, 2),
        "%"
    )

    print(
        "GT items:",
        gt_items
    )

    print(
        "Predicted items:",
        pred_items
    )

    print(
        "Matched items:",
        matched_items
    )


# ============================================
# FINAL COMPARISON
# ============================================

print("\n\n============================================")
print("FINAL RESOLUTION COMPARISON")
print("============================================")


print(
    f"{'Resolution':<15}"
    f"{'Avg Time':<12}"
    f"{'VRAM':<10}"
    f"{'JSON':<10}"
    f"{'Total':<10}"
    f"{'Name':<10}"
    f"{'Qty':<10}"
    f"{'Price':<10}"
    f"{'Complete'}"
)

print("-" * 105)


for result in results:

    print(

        f"{result['resolution']:<15}"

        f"{result['average_time']:.2f}s"
        f"{'':<7}"

        f"{result['peak_memory']:.2f}GB"
        f"{'':<5}"

        f"{result['json_accuracy']:.1f}%"
        f"{'':<5}"

        f"{result['total_accuracy']:.1f}%"
        f"{'':<5}"

        f"{result['name_accuracy']:.1f}%"
        f"{'':<5}"

        f"{result['quantity_accuracy']:.1f}%"
        f"{'':<5}"

        f"{result['price_accuracy']:.1f}%"
        f"{'':<5}"

        f"{result['complete_accuracy']:.1f}%"

    )


# ============================================
# BEST RESOLUTION BY COMPLETE ACCURACY
# ============================================

best_accuracy = max(

    results,

    key=lambda x:
    x["complete_accuracy"]

)


# ============================================
# FASTEST RESOLUTION
# ============================================

fastest_resolution = min(

    results,

    key=lambda x:
    x["average_time"]

)


print("\n============================================")

print(
    "BEST ACCURACY:",
    best_accuracy["resolution"]
)

print(
    "Complete item accuracy:",
    round(
        best_accuracy["complete_accuracy"],
        2
    ),
    "%"
)


print("\nFASTEST RESOLUTION:")

print(
    fastest_resolution["resolution"]
)

print(
    "Average time:",
    round(
        fastest_resolution["average_time"],
        2
    ),
    "sec"
)


# ============================================
# FINAL MESSAGE
# ============================================

print("\n============================================")
print("PHASE 8.2B TEST COMPLETE")
print("============================================")

print(
    "\nUse the accuracy + speed + VRAM results "
    "together to select the final resolution."
)