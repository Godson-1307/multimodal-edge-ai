import json
import time
import torch

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

NUM_SAMPLES = 5

MAX_IMAGE_SIZE = 768


# ============================================================
# STEP 1: LOAD CORD VALIDATION DATASET
# ============================================================

dataset = load_dataset(
    "naver-clova-ix/cord-v2",
    split="validation"
)

eval_dataset = dataset.select(range(NUM_SAMPLES))

print("========== PHASE 7: EVALUATION ==========")
print("Validation samples available:", len(dataset))
print("Samples selected:", NUM_SAMPLES)


# ============================================================
# STEP 2: CORD TEXT EXTRACTION
# ============================================================

def get_text(annotation):

    words = annotation.get("words", [])

    text = " ".join(
        word.get("text", "").strip()
        for word in words
        if word.get("text", "").strip()
    )

    return text.strip()


# ============================================================
# STEP 3: FIND CORD ANNOTATIONS
# ============================================================

def find_annotations(data):

    if isinstance(data, list):

        # Check if this is the actual annotation list
        if data and all(
            isinstance(item, dict) and "category" in item
            for item in data
        ):
            return data

        # Search recursively
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


# ============================================================
# STEP 4: EXTRACT GROUND TRUTH
# ============================================================

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


    # --------------------------------------------------------
    # Process annotations
    # --------------------------------------------------------

    for annotation in annotations:

        category = annotation.get("category")
        group_id = annotation.get("group_id")

        text = get_text(annotation)


        # ----------------------------------------------------
        # LINE ITEMS
        # ----------------------------------------------------

        if category and category.startswith("menu."):

            if group_id not in menu_items:

                menu_items[group_id] = {
                    "name": None,
                    "quantity": None,
                    "price": None
                }


            # Item name
            if category == "menu.nm":

                menu_items[group_id]["name"] = text


            # Quantity
            elif category == "menu.cnt":

                parts = text.split()

                if parts:

                    menu_items[group_id]["quantity"] = parts[0]


            # Price
            elif category == "menu.price":

                menu_items[group_id]["price"] = text


        # ----------------------------------------------------
        # TOTAL TAX
        # ----------------------------------------------------

        elif category == "sub_total.tax_price":

            target["total_tax"] = text


        # ----------------------------------------------------
        # TOTAL AMOUNT
        # ----------------------------------------------------

        elif category == "total.total_price":

            target["total_amount"] = text


    # --------------------------------------------------------
    # Convert menu dictionary to list
    # --------------------------------------------------------

    for group_id in sorted(menu_items):

        item = menu_items[group_id]

        target["line_items"].append({
            "name": item["name"],
            "quantity": item["quantity"],
            "price": item["price"]
        })


    return target


# ============================================================
# STEP 5: JSON PREDICTION PARSER
# ============================================================

def parse_prediction(prediction_text):

    prediction_text = prediction_text.strip()


    # --------------------------------------------------------
    # Remove Markdown JSON code block
    # --------------------------------------------------------

    if prediction_text.startswith("```json"):

        prediction_text = prediction_text[7:]

    elif prediction_text.startswith("```"):

        prediction_text = prediction_text[3:]


    if prediction_text.endswith("```"):

        prediction_text = prediction_text[:-3]


    prediction_text = prediction_text.strip()


    # --------------------------------------------------------
    # Convert text to JSON
    # --------------------------------------------------------

    try:

        prediction = json.loads(prediction_text)

        return prediction, True

    except json.JSONDecodeError:

        return None, False


# ============================================================
# STEP 6: RESIZE IMAGE
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
# STEP 7: LOAD BASE QWEN MODEL
# ============================================================

print("\n========== LOADING BASE MODEL ==========")


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


model.eval()


print("Base model loaded successfully.")


# ============================================================
# GPU INFORMATION
# ============================================================

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU memory allocated:",
        round(
            torch.cuda.memory_allocated() / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# STEP 8: GENERATE BASE MODEL PREDICTION
# ============================================================

def generate_prediction(image):

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


    # Create Qwen input format
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


    # Move tensors to GPU
    inputs = {
        key: value.to(model.device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }


    # Start timer
    start_time = time.time()


    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512
        )


    inference_time = time.time() - start_time


    # Remove input tokens
    generated_ids = output_ids[
        :,
        inputs["input_ids"].shape[1]:
    ]


    # Decode generated text
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()


    return output_text, inference_time


# ============================================================
# STEP 9: BASE MODEL EVALUATION
# ============================================================

print("\n========== BASE MODEL EVALUATION ==========")


json_valid_count = 0


for i, sample in enumerate(eval_dataset):

    print(f"\n----- SAMPLE {i + 1} -----")


    image = sample["image"]


    # Get ground truth
    ground_truth = extract_ground_truth(
        sample["ground_truth"]
    )


    print(
        "Image:",
        image.size
    )


    print(
        "Ground Truth Total:",
        ground_truth["total_amount"]
    )


    print(
        "Ground Truth Line Items:",
        len(ground_truth["line_items"])
    )


    try:

        # Generate prediction
        prediction_text, inference_time = generate_prediction(
            image
        )


        # ----------------------------------------------------
        # Raw prediction
        # ----------------------------------------------------

        print("\nBase Model Raw Prediction:")

        print(prediction_text)


        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        prediction, json_valid = parse_prediction(
            prediction_text
        )


        print(
            "\nJSON Valid:",
            json_valid
        )


        if json_valid:

            json_valid_count += 1


            print("\nParsed Prediction:")

            print(
                json.dumps(
                    prediction,
                    indent=2,
                    ensure_ascii=False
                )
            )

        else:

            print(
                "Could not parse prediction as JSON."
            )


        # ----------------------------------------------------
        # Inference time
        # ----------------------------------------------------

        print(
            "\nInference Time:",
            round(inference_time, 2),
            "seconds"
        )


    except Exception as e:

        print("\nERROR:", e)


# ============================================================
# STEP 10: JSON VALIDITY RESULT
# ============================================================

json_validity = (
    json_valid_count / NUM_SAMPLES
) * 100


print("\n========== JSON VALIDITY RESULT ==========")

print(
    "Valid JSON:",
    f"{json_valid_count}/{NUM_SAMPLES}"
)

print(
    "JSON Validity Rate:",
    f"{json_validity:.2f}%"
)


# ============================================================
# FINAL
# ============================================================

print("\n============================================")
print("STEP 7.1C SUCCESS")
print("STEP 7.1D SUCCESS")
print("Base model JSON validation completed.")
print("============================================")