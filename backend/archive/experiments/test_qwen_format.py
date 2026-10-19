import json
from datasets import load_dataset
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Load CORD
dataset = load_dataset("naver-clova-ix/cord-v2")

# Load our processed target
with open("data/processed/train.jsonl", "r", encoding="utf-8") as f:
    record = json.loads(f.readline())

image_id = record["image_id"]
target = record["target"]

# Get image
image = dataset["train"][image_id]["image"]

prompt = """
Extract the following information from this receipt:

- vendor_name
- invoice_number
- invoice_date
- gstin
- state
- state_code
- total_tax
- total_amount
- line_items

Return ONLY a valid JSON object.
If a field is not visible, use null.
"""

# Create Qwen conversation
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
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": json.dumps(target, ensure_ascii=False)
            }
        ]
    }
]

# Apply Qwen chat template
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

print("\n========== QWEN TRAINING FORMAT ==========")
print(text)

print("\n========== TARGET JSON ==========")
print(json.dumps(target, indent=2, ensure_ascii=False))

print("\n===========================================")
print("Qwen training format created successfully!")