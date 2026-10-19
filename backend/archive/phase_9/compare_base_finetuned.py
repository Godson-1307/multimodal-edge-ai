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


print("\n========== BASE VS FINE-TUNED ==========")


# --------------------------------------------------
# 1. Processor
# --------------------------------------------------

processor = AutoProcessor.from_pretrained(MODEL_ID)


# --------------------------------------------------
# 2. Quantization
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

print("\nLoading base model...")

base_model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

base_model.eval()


# --------------------------------------------------
# 4. Load LoRA model
# --------------------------------------------------

print("Loading LoRA adapter...")

finetuned_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

finetuned_model.eval()

print("Fine-tuned model loaded.")


# --------------------------------------------------
# 5. Validation sample
# --------------------------------------------------

dataset = CORDTrainingDataset("validation")

sample = dataset[0]

image = sample["image"]
ground_truth = sample["target"]


# Resize image
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


# --------------------------------------------------
# 7. Generate function
# --------------------------------------------------

def generate_prediction(model):

    model_device = next(model.parameters()).device

    model_inputs = {}

    for key, value in inputs.items():

        if torch.is_tensor(value):
            model_inputs[key] = value.to(model_device)

        else:
            model_inputs[key] = value

    with torch.no_grad():

        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=300,
            do_sample=False
        )

    input_length = model_inputs["input_ids"].shape[1]

    generated_ids = output_ids[:, input_length:]

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return response


# --------------------------------------------------
# 8. Base prediction
# --------------------------------------------------

print("\nGenerating BASE prediction...")

base_prediction = generate_prediction(base_model)


# --------------------------------------------------
# 9. Fine-tuned prediction
# --------------------------------------------------

print("Generating FINE-TUNED prediction...")

finetuned_prediction = generate_prediction(
    finetuned_model
)


# --------------------------------------------------
# 10. Results
# --------------------------------------------------

print("\n========== GROUND TRUTH ==========")

print(
    json.dumps(
        ground_truth,
        indent=2,
        ensure_ascii=False
    )
)


print("\n========== BASE MODEL ==========")

print(base_prediction)


print("\n========== FINE-TUNED MODEL ==========")

print(finetuned_prediction)


print("\n========== PHASE 6 RESULT ==========")

print("QLoRA adapter loading: SUCCESS")
print("Fine-tuned inference: SUCCESS")
print("Base vs fine-tuned comparison: SUCCESS")

print("\nPHASE 6 COMPLETE")
print("====================================")