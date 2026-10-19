from datasets import load_dataset

print("Loading CORD...")

dataset = load_dataset("naver-clova-ix/cord-v2")

print("\nDataset loaded!")

print(dataset)

print("\n--- FIRST TRAINING SAMPLE ---")

sample = dataset["train"][0]

print("Keys:", sample.keys())

print("\nGround truth:")
print(sample["ground_truth"])

print("\nImage:")
print(sample["image"])