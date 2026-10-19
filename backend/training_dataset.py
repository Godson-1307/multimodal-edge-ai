import json
from datasets import load_dataset


class CORDTrainingDataset:

    def __init__(self, split="train"):
        self.dataset = load_dataset("naver-clova-ix/cord-v2")[split]

    def __len__(self):
        return len(self.dataset)

    def get_text(self, annotation):
        """
        Combine all OCR words belonging to one annotation.
        """

        words = annotation.get("words", [])

        text = " ".join(
            word.get("text", "").strip()
            for word in words
            if word.get("text", "").strip()
        )

        return text.strip()

    def find_annotations(self, data):
        """
        Find the list containing annotation dictionaries.
        """

        if isinstance(data, list):

            # Check whether this is the annotation list
            if data and all(
                isinstance(item, dict) and "category" in item
                for item in data
            ):
                return data

            # Otherwise search deeper
            for item in data:

                result = self.find_annotations(item)

                if result is not None:
                    return result

        elif isinstance(data, dict):

            for value in data.values():

                result = self.find_annotations(value)

                if result is not None:
                    return result

        return None

    def __getitem__(self, index):

        sample = self.dataset[index]

        image = sample["image"]

        ground_truth = json.loads(sample["ground_truth"])

        # Find actual CORD annotation list
        annotations = self.find_annotations(ground_truth)

        if annotations is None:
            raise ValueError(
                "Could not find CORD annotation list."
            )

        target = {
            "vendor_name": None,
            "invoice_number": None,
            "invoice_date": None,
            "gstin": None,
            "state": None,
            "state_code": None,
            "total_tax": None,
            "total_amount": None,
            "line_items": []
        }

        # Store menu fields using group_id
        menu_items = {}

        for annotation in annotations:

            category = annotation.get("category")
            group_id = annotation.get("group_id")

            text = self.get_text(annotation)

            # -------------------------
            # MENU ITEMS
            # -------------------------

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

                    # Example:
                    # "1 x" -> "1"

                    parts = text.split()

                    if parts:
                        menu_items[group_id]["quantity"] = parts[0]

                elif category == "menu.price":

                    menu_items[group_id]["price"] = text

            # -------------------------
            # TOTAL TAX
            # -------------------------

            elif category == "sub_total.tax_price":

                target["total_tax"] = text

            # -------------------------
            # TOTAL AMOUNT
            # -------------------------

            elif category == "total.total_price":

                target["total_amount"] = text

        # -------------------------
        # CREATE LINE ITEMS
        # -------------------------

        for group_id in sorted(menu_items):

            item = menu_items[group_id]

            target["line_items"].append({
                "name": item["name"],
                "quantity": item["quantity"],
                "price": item["price"]
            })

        return {
            "image": image,
            "target": target
        }


# ==========================================
# TEST DATASET
# ==========================================

if __name__ == "__main__":

    dataset = CORDTrainingDataset("train")

    print("\n========== DATASET TEST ==========")

    print("Number of samples:", len(dataset))

    for i in range(3):

        sample = dataset[i]

        print("\nSample:", i)

        print("Image size:", sample["image"].size)

        print("\nTarget:")

        print(
            json.dumps(
                sample["target"],
                indent=2,
                ensure_ascii=False
            )
        )

        print(
            "\nLine items:",
            len(sample["target"]["line_items"])
        )

    print("\nSUCCESS: Dataset loader works!")

    print("=================================")