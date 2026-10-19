import json
import torch

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig
)

from peft import PeftModel

from training_dataset import CORDTrainingDataset


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "outputs/qlora_mini"


print("\n========== FINE-TUNED MODEL TEST ==========")

# --------------------------------------------------
# 1. Load processor
# --------------------------------------------------

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(MODEL_ID)


# --------------------------------------------------
# 2. 4-bit configuration
# --------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)


# --------------------------------------------------
# 3. Load base model
# --------------------------------------------------

print("Loading base model...")

base_model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)


# --------------------------------------------------
# 4. Load LoRA adapter
# --------------------------------------------------

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("LoRA adapter loaded successfully.")


# --------------------------------------------------
# 5. Load validation sample
# --------------------------------------------------

print("\nLoading CORD validation sample...")

dataset = CORDTrainingDataset("validation")

sample = dataset[0]

image = sample["image"]
ground_truth = sample["target"]

print("Image size:", image.size)

# Keep image size controlled
max_size = 768

width, height = image.size

if max(width, height) > max_size:
    scale = max_size / max(width, height)

    image = image.resize(
        (
            int(width * scale),
            int(height * scale)
        )
    )

print("Resized image:", image.size)


# --------------------------------------------------
# 6. Prompt
# --------------------------------------------------

prompt = """
Extract the information from this receipt/document.

Return ONLY a valid JSON object with these fields:

- vendor_name
- invoice_number
- invoice_date
- gstin
- state
- state_code
- total_tax
- total_amount
- line_items

For line_items use:

[
    {
        "name": "...",
        "quantity": "...",
        "price": "..."
    }
]

If a field is not visible, use null.

Do not add explanations.
Do not use markdown.
"""


# --------------------------------------------------
# 7. Prepare chat
# --------------------------------------------------

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


# Move tensors to model device
model_device = next(model.parameters()).device

for key in inputs:
    if torch.is_tensor(inputs[key]):
        inputs[key] = inputs[key].to(model_device)


# --------------------------------------------------
# 8. Generate
# --------------------------------------------------

print("\nGenerating prediction...")

with torch.no_grad():

    output_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False
    )


# --------------------------------------------------
# 9. Decode only generated tokens
# --------------------------------------------------

input_length = inputs["input_ids"].shape[1]

generated_ids = output_ids[:, input_length:]

response = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\n========== GROUND TRUTH ==========")

print(
    json.dumps(
        ground_truth,
        indent=2,
        ensure_ascii=False
    )
)


print("\n========== FINE-TUNED PREDICTION ==========")

print(response)


print("\n========== TEST COMPLETE ==========")
print("Base model + LoRA adapter inference: SUCCESS")
print("====================================")