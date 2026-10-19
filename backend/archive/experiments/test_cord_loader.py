import json
from datasets import load_dataset

# Load CORD dataset
dataset = load_dataset("naver-clova-ix/cord-v2")

# Load our processed training file
with open("data/processed/train.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

# Test only first 2 samples
for record in records[:2]:

    image_id = record["image_id"]
    target = record["target"]

    # Get corresponding image from CORD
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

    print("\n==============================")
    print("IMAGE ID:", image_id)
    print("IMAGE SIZE:", image.size)

    print("\nPROMPT:")
    print(prompt.strip())

    print("\nTARGET:")
    print(json.dumps(target, indent=2, ensure_ascii=False))