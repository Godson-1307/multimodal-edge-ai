import json
from pathlib import Path


INPUT_FILE = Path("data/processed/train.jsonl")
OUTPUT_FILE = Path("data/processed/train_improved.jsonl")


def clean_value(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


def clean_item(item):
    return {
        "name": clean_value(item.get("name")),
        "quantity": clean_value(item.get("quantity")),
        "price": clean_value(item.get("price"))
    }


def improve_target(target):
    line_items = []

    for item in target.get("line_items", []):
        cleaned = clean_item(item)

        # Keep only meaningful items
        if cleaned["name"] is not None:
            line_items.append(cleaned)

    improved = {
        "total_amount": clean_value(
            target.get("total_amount")
        ),
        "total_tax": clean_value(
            target.get("total_tax")
        ),
        "line_items": line_items
    }

    return improved


print("=" * 70)
print("PHASE 9.4 - DATASET / TARGET IMPROVEMENT")
print("=" * 70)

count = 0
items_before = 0
items_after = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

    for line in f_in:

        record = json.loads(line)

        target = record["target"]

        items_before += len(target.get("line_items", []))

        improved_target = improve_target(target)

        items_after += len(
            improved_target["line_items"]
        )

        new_record = {
            "image_id": record["image_id"],
            "target": improved_target
        }

        f_out.write(
            json.dumps(
                new_record,
                ensure_ascii=False
            ) + "\n"
        )

        count += 1


print()
print("Samples processed:", count)
print("Items before:", items_before)
print("Items after:", items_after)

print()
print("Example improved target:")

with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    example = json.loads(f.readline())

print(
    json.dumps(
        example,
        indent=2,
        ensure_ascii=False
    )
)

print()
print("Saved to:", OUTPUT_FILE)
print()
print("PHASE 9.4 DATA PREPARATION COMPLETE")
print("=" * 70)