import json
import torch
from datasets import load_dataset
from transformers import AutoProcessor


# -----------------------------
# 1. Model
# -----------------------------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


# -----------------------------
# 2. Load CORD
# -----------------------------
dataset = load_dataset("naver-clova-ix/cord-v2")

sample = dataset["train"][0]

image = sample["image"]
ground_truth = json.loads(sample["ground_truth"])


# -----------------------------
# 3. Convert target to JSON
# -----------------------------
target = {
    "vendor_name": None,
    "invoice_number": None,
    "invoice_date": None,
    "gstin": None,
    "state": None,
    "state_code": None,
    "total_tax": None,
    "total_amount": ground_truth.get("total", {}).get("total_price"),
    "line_items": []
}


# Add menu items
for item in ground_truth.get("menu", []):
    target["line_items"].append({
        "name": item.get("nm"),
        "quantity": item.get("cnt"),
        "price": item.get("price")
    })


target_json = json.dumps(target, ensure_ascii=False)


# -----------------------------
# 4. Prompt
# -----------------------------
prompt = """
Extract the information from this receipt.

Return ONLY a valid JSON object using this structure:

{
    "vendor_name": null,
    "invoice_number": null,
    "invoice_date": null,
    "gstin": null,
    "state": null,
    "state_code": null,
    "total_tax": null,
    "total_amount": null,
    "line_items": []
}

If information is not available, use null.
For line_items, use name, quantity and price.
"""


# -----------------------------
# 5. Create Qwen conversation
# -----------------------------
conversation = [
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
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": target_json
            }
        ]
    }
]


# -----------------------------
# 6. Load processor
# -----------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)


# -----------------------------
# 7. Apply chat template
# -----------------------------
text = processor.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=False
)


# -----------------------------
# 8. Process image + text
# -----------------------------
inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt"
)


# -----------------------------
# 9. Display results
# -----------------------------
print("\n========== TRAINING SAMPLE TEST ==========")

print("\nImage:")
print("Type:", type(image))
print("Size:", image.size)

print("\nTarget JSON:")
print(target_json)

print("\nProcessor inputs:")
for key, value in inputs.items():
    if hasattr(value, "shape"):
        print(key, value.shape)
    else:
        print(key, type(value))

print("\nInput IDs:", inputs["input_ids"].shape)

print("\nSUCCESS: One CORD training sample was prepared!")

print("==========================================")