import json
import re
import time

import torch
from datasets import load_dataset
from PIL import Image

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

VALIDATION_FILE = "data/processed/validation.jsonl"

MAX_IMAGE_WIDTH = 384
MAX_IMAGE_HEIGHT = 576
MAX_NEW_TOKENS = 128

NUM_SAMPLES = 5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value).strip().lower()

    value = re.sub(r"\s+", " ", value)

    return value


def normalize_number(value):

    if value is None:
        return ""

    value = str(value)

    value = re.sub(r"[,\s₹$]", "", value)

    return value.strip()


def parse_json_response(text):

    text = text.strip()

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "").strip()

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

        try:

            return json.loads(match.group())

        except Exception:

            pass


    return None


def resize_image(image):

    image = image.copy()

    image.thumbnail(
        (
            MAX_IMAGE_WIDTH,
            MAX_IMAGE_HEIGHT
        ),
        Image.Resampling.LANCZOS
    )

    return image


# ============================================================
# START
# ============================================================

print("=" * 60)
print("PHASE 8.10B - EVALUATION VALIDATION")
print("=" * 60)


# ============================================================
# LOAD PROCESSED GROUND TRUTH
# ============================================================

print("\nLoading processed validation JSONL...")

records = []

with open(
    VALIDATION_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if line.strip():

            records.append(
                json.loads(line)
            )


print(
    "Processed validation records:",
    len(records)
)


# ============================================================
# LOAD CORD DATASET
# ============================================================

print("\nLoading CORD v2 dataset...")

dataset = load_dataset(
    "naver-clova-ix/cord-v2"
)

validation_images = dataset["validation"]

print(
    "CORD validation images:",
    len(validation_images)
)


# ============================================================
# CHECK ALIGNMENT
# ============================================================

print("\n========== DATA ALIGNMENT CHECK ==========")

print(
    "Processed record 0 image_id:",
    records[0]["image_id"]
)

print(
    "CORD validation image 0 type:",
    type(validation_images[0]["image"])
)

print(
    "CORD validation image 0 size:",
    validation_images[0]["image"].size
)

print("\nAlignment assumption:")
print("processed JSONL index == CORD validation index")


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


model = AutoModelForMultimodalLM.from_pretrained(

    MODEL_ID,

    quantization_config=bnb_config,

    device_map="auto"
)


print("Base model loaded.")


# ============================================================
# LOAD QLORA
# ============================================================

print("\nLoading QLoRA adapter...")


model = PeftModel.from_pretrained(

    model,

    ADAPTER_PATH
)


model.eval()


print("QLoRA adapter loaded.")


# ============================================================
# PROCESSOR
# ============================================================

processor = AutoProcessor.from_pretrained(
    ADAPTER_PATH
)


print("Processor loaded.")


# ============================================================
# DEVICE
# ============================================================

device = next(
    model.parameters()
).device


print("\nModel device:", device)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# METRICS
# ============================================================

json_correct = 0

total_gt_items = 0
total_pred_items = 0
total_matched_items = 0

times = []


# ============================================================
# RUN VALIDATION
# ============================================================

print("\n")
print("=" * 60)
print("RUNNING VALIDATION")
print("=" * 60)


for index in range(NUM_SAMPLES):


    print("\n")
    print("-" * 60)

    print(
        f"SAMPLE {index + 1}/{NUM_SAMPLES}"
    )

    print("-" * 60)


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    record = records[index]

    image_id = record["image_id"]

    ground_truth = record["target"]


    print(
        "Image ID:",
        image_id
    )


    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image = validation_images[index]["image"]

    image = image.convert("RGB")


    print(
        "Original image:",
        image.size
    )


    image = resize_image(image)


    print(
        "Resized image:",
        image.size
    )


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = """
Extract the invoice information from the image.

Return ONLY a valid JSON object.

Use this structure:

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

If a field is not visible, use null.
Do not add explanations.
Do not use markdown.
"""


    # --------------------------------------------------------
    # Chat template
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

        padding=True,

        return_tensors="pt"

    )


    inputs = {

        key: value.to(device)
        if hasattr(value, "to")
        else value

        for key, value in inputs.items()

    }


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()


    start = time.time()


    with torch.inference_mode():

        outputs = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            use_cache=True

        )


    elapsed = time.time() - start


    times.append(elapsed)


    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    input_length = inputs["input_ids"].shape[1]


    generated_tokens = outputs[
        :,
        input_length:
    ]


    response = processor.batch_decode(

        generated_tokens,

        skip_special_tokens=True

    )[0]


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    prediction = parse_json_response(
        response
    )


    if prediction is not None:

        json_correct += 1

        json_status = "YES"

    else:

        json_status = "NO"


    # --------------------------------------------------------
    # Print GT
    # --------------------------------------------------------

    print("\nGROUND TRUTH:")

    print(
        json.dumps(
            ground_truth,
            indent=2,
            ensure_ascii=False
        )
    )


    # --------------------------------------------------------
    # Print raw prediction
    # --------------------------------------------------------

    print("\nRAW MODEL OUTPUT:")

    print(response)


    # --------------------------------------------------------
    # Print parsed prediction
    # --------------------------------------------------------

    print("\nPARSED PREDICTION:")


    if prediction is not None:

        print(
            json.dumps(
                prediction,
                indent=2,
                ensure_ascii=False
            )
        )

    else:

        print("INVALID JSON")


    print(
        "\nJSON VALID:",
        json_status
    )


    print(
        "Inference time:",
        round(elapsed, 2),
        "seconds"
    )


    # --------------------------------------------------------
    # Line item metrics
    # --------------------------------------------------------

    if prediction is None:

        continue


    gt_items = ground_truth.get(
        "line_items",
        []
    )


    pred_items = prediction.get(
        "line_items",
        []
    )


    total_gt_items += len(gt_items)

    total_pred_items += len(pred_items)


    matched = 0


    for gt_item in gt_items:

        gt_name = normalize_text(
            gt_item.get("name")
        )


        for pred_item in pred_items:

            pred_name = normalize_text(
                pred_item.get("name")
            )


            if (
                gt_name
                and
                gt_name == pred_name
            ):

                matched += 1

                break


    total_matched_items += matched


    print("\nLINE ITEMS")

    print(
        "Ground truth:",
        len(gt_items)
    )

    print(
        "Predicted:",
        len(pred_items)
    )

    print(
        "Matched:",
        matched
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("PHASE 8.10B RESULTS")
print("=" * 60)


tested = NUM_SAMPLES


print(
    "\nValid JSON:",
    f"{json_correct}/{tested}",
    f"({json_correct / tested * 100:.2f}%)"
)


print("\nLINE ITEM CHECK")

print(
    "GT items:",
    total_gt_items
)

print(
    "Predicted items:",
    total_pred_items
)

print(
    "Matched items:",
    total_matched_items
)


# ------------------------------------------------------------
# Precision
# ------------------------------------------------------------

if total_pred_items > 0:

    precision = (
        total_matched_items /
        total_pred_items
    ) * 100

else:

    precision = 0


# ------------------------------------------------------------
# Recall
# ------------------------------------------------------------

if total_gt_items > 0:

    recall = (
        total_matched_items /
        total_gt_items
    ) * 100

else:

    recall = 0


# ------------------------------------------------------------
# F1
# ------------------------------------------------------------

if precision + recall > 0:

    f1 = (
        2 * precision * recall /
        (precision + recall)
    )

else:

    f1 = 0


print(
    "\nPrecision:",
    round(precision, 2),
    "%"
)

print(
    "Recall:",
    round(recall, 2),
    "%"
)

print(
    "F1:",
    round(f1, 2),
    "%"
)


# ============================================================
# PERFORMANCE
# ============================================================

if times:

    print("\nPERFORMANCE")

    print(
        "Average:",
        round(
            sum(times) / len(times),
            2
        ),
        "seconds"
    )

    print(
        "Fastest:",
        round(
            min(times),
            2
        ),
        "seconds"
    )

    print(
        "Slowest:",
        round(
            max(times),
            2
        ),
        "seconds"
    )


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    peak = (
        torch.cuda.max_memory_allocated()
        /
        (1024 ** 3)
    )

    allocated = (
        torch.cuda.memory_allocated()
        /
        (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        /
        (1024 ** 3)
    )


    print("\nGPU MEMORY")

    print(
        "Peak:",
        round(peak, 2),
        "GB"
    )

    print(
        "Allocated:",
        round(allocated, 2),
        "GB"
    )

    print(
        "Reserved:",
        round(reserved, 2),
        "GB"
    )


print("\n")
print("PHASE 8.10B COMPLETE")
print("=" * 60)