import json
from datasets import load_dataset


dataset = load_dataset("naver-clova-ix/cord-v2")

sample = dataset["train"][0]

ground_truth = json.loads(sample["ground_truth"])

print("\n========== RAW CORD SAMPLE ==========")

print("\nTop-level keys:")
print(ground_truth.keys())

print("\nFull ground truth:")
print(json.dumps(
    ground_truth,
    indent=2,
    ensure_ascii=False
))

print("\n====================================")