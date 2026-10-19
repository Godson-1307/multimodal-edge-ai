import json
import os
from datasets import load_dataset


print("Loading CORD dataset...")

dataset = load_dataset("naver-clova-ix/cord-v2")

print("Dataset loaded!")


# =====================================
# CLEANING FUNCTIONS
# =====================================

def clean_quantity(value):
    if value is None:
        return None

    return str(value).replace("x", "").strip()


def clean_price(value):
    if value is None:
        return None

    return str(value).replace(",", "").replace(".", "").strip()


# =====================================
# CONVERT MENU ITEM
# =====================================

def convert_menu_item(item):

    # Normal CORD format:
    #
    # {
    #     "nm": "...",
    #     "cnt": "...",
    #     "price": "..."
    # }

    if not isinstance(item, dict):
        return None

    name = item.get("nm")
    quantity = item.get("cnt")
    price = item.get("price")

    # If this is not a useful menu item
    if name is None and quantity is None and price is None:
        return None

    return {
        "name": name,
        "quantity": clean_quantity(quantity),
        "price": clean_price(price)
    }


# =====================================
# CONVERT ONE CORD SAMPLE
# =====================================

def convert_cord_sample(sample, image_id):

    ground_truth = sample["ground_truth"]

    # Ground truth is stored as a JSON string
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    gt_parse = ground_truth.get("gt_parse", {})

    # =================================
    # LINE ITEMS
    # =================================

    menu = gt_parse.get("menu", [])

    line_items = []

    # CORD normally stores menu as a list
    if isinstance(menu, list):

        for item in menu:

            converted_item = convert_menu_item(item)

            if converted_item is not None:
                line_items.append(converted_item)

    # Some malformed/unexpected samples may contain
    # another structure. We don't crash the entire dataset.
    elif isinstance(menu, dict):

        converted_item = convert_menu_item(menu)

        if converted_item is not None:
            line_items.append(converted_item)

    else:

        print(
            f"Warning: unexpected menu type "
            f"{type(menu)} in image_id {image_id}"
        )

    # =================================
    # SUBTOTAL / TAX
    # =================================

    sub_total = gt_parse.get("sub_total", {})

    if not isinstance(sub_total, dict):
        sub_total = {}

    tax_value = sub_total.get("tax_price")

    if tax_value:
        total_tax = clean_price(tax_value)
    else:
        total_tax = None

    # =================================
    # TOTAL
    # =================================

    total = gt_parse.get("total", {})

    if not isinstance(total, dict):
        total = {}

    total_value = total.get("total_price")

    if total_value:
        total_amount = clean_price(total_value)
    else:
        total_amount = None

    # =================================
    # NORMALIZED TARGET
    # =================================

    normalized_data = {

        "vendor_name": None,

        "invoice_number": None,

        "invoice_date": None,

        "gstin": None,

        "state": None,

        "state_code": None,

        "total_tax": total_tax,

        "total_amount": total_amount,

        "line_items": line_items
    }

    # =================================
    # TRAINING RECORD
    # =================================

    record = {

        "image_id": image_id,

        "target": normalized_data
    }

    return record


# =====================================
# SAVE DATASET SPLIT
# =====================================

def save_split(split_name):

    samples = dataset[split_name]

    output_path = f"data/processed/{split_name}.jsonl"

    print("\n==============================")
    print(f"Processing {split_name}...")
    print("==============================")

    print(f"Number of samples: {len(samples)}")

    processed_count = 0

    skipped_count = 0

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for image_id, sample in enumerate(samples):

            try:

                record = convert_cord_sample(
                    sample,
                    image_id
                )

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                processed_count += 1

            except Exception as error:

                skipped_count += 1

                print(
                    f"\nWarning: failed to process "
                    f"image_id {image_id}"
                )

                print("Error:", error)

    print("\n------------------------------")
    print(f"Processed: {processed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Saved:     {output_path}")
    print("------------------------------")


# =====================================
# CREATE OUTPUT DIRECTORY
# =====================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# =====================================
# PROCESS ALL SPLITS
# =====================================

save_split("train")

save_split("validation")

save_split("test")


# =====================================
# FINISHED
# =====================================

print("\n==============================")
print("DATASET PROCESSING COMPLETE")
print("==============================")