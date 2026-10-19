import json
from datasets import load_dataset


print("=" * 60)
print("CORD GROUND TRUTH DIAGNOSTIC")
print("=" * 60)

print("\nLoading validation dataset...")

dataset = load_dataset(
    "naver-clova-ix/cord-v2",
    split="validation"
)

print("Validation samples:", len(dataset))


for i in range(5):

    sample = dataset[i]

    print("\n" + "=" * 60)
    print(f"SAMPLE {i + 1}")
    print("=" * 60)

    print("Image size:", sample["image"].size)

    raw_gt = sample["ground_truth"]

    print("\nGround truth type:")
    print(type(raw_gt))

    # Parse JSON
    if isinstance(raw_gt, str):

        try:
            raw_gt = json.loads(raw_gt)
        except Exception as e:
            print("JSON parsing error:", e)
            continue

    print("\nTop-level keys:")
    print(raw_gt.keys())

    gt_parse = raw_gt.get("gt_parse", {})

    print("\ngt_parse keys:")
    print(gt_parse.keys())

    # ----------------------------------------
    # MENU
    # ----------------------------------------

    menu = gt_parse.get("menu", [])

    print("\nMENU TYPE:")
    print(type(menu))

    if isinstance(menu, list):

        print("Number of menu items:", len(menu))

        for j, item in enumerate(menu[:5]):

            print(f"\nMenu item {j + 1}:")
            print(item)

    else:

        print("Menu content:")
        print(menu)

    # ----------------------------------------
    # TOTAL
    # ----------------------------------------

    total = gt_parse.get("total", {})

    print("\nTOTAL:")
    print(total)

    # ----------------------------------------
    # SUB TOTAL
    # ----------------------------------------

    subtotal = gt_parse.get("sub_total", {})

    print("\nSUB TOTAL:")
    print(subtotal)


print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
