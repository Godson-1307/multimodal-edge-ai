import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig
)

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("Loading model...")

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)

print("Loading invoice...")
image = Image.open("data/samples/invoice.jpg")

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
                "text": "Describe this document and identify the important information visible in it."
            }
        ]
    }
]

print("Preparing input...")

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

inputs = inputs.to(model.device)

print("Generating response...")

outputs = model.generate(
    **inputs,
    max_new_tokens=200
)

response = processor.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print("\nMODEL RESPONSE:")
print(response)
