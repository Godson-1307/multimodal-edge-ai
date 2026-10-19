import torch
from PIL import Image
from json_utils import parse_json_response
from validation import validate_extraction
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig
)


def show_gpu_memory(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"\n--- GPU MEMORY: {stage} ---")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
    else:
        print("\nCUDA is not available.")


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

show_gpu_memory("Before model loading")

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
    dtype=torch.float16,
    quantization_config=quantization_config
)

show_gpu_memory("After model loading")

torch.cuda.reset_peak_memory_stats()

print("\n--- CUDA DEVICE ---")
print(torch.cuda.get_device_name(0))

print("\n--- MODEL DEVICE CHECK ---")

devices = {}

for name, param in model.named_parameters():
    device = str(param.device)
    devices[device] = devices.get(device, 0) + param.numel()

for device, count in devices.items():
    print(f"{device}: {count:,} parameters")


print("Loading invoice...")
image = Image.open("data/samples/invoice.jpg")

prompt = """
Extract the following information from this invoice:

Return ONLY valid JSON.

Do not use Markdown.
Do not include ```json.
Do not include explanations.

Use exactly these fields:

{
    "vendor_name": null,
    "invoice_number": null,
    "invoice_date": null,
    "gstin": null,
    "state": null,
    "state_code": null,
    "total_tax": null,
    "total_amount": null
}

If a field cannot be found, use null.
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
    max_new_tokens=300
)

show_gpu_memory("After generation")

peak = torch.cuda.max_memory_allocated() / 1024**3

print(f"\nPeak GPU memory allocated: {peak:.2f} GB")

response = processor.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print("\nRAW MODEL RESPONSE:")
print(response)

try:
    data = parse_json_response(response)

    print("\n==============================")
    print("EXTRACTED INVOICE DATA")
    print("==============================")

    for field, value in data.items():
        print(f"{field}: {value}")

    missing = validate_extraction(data)

    print("\nVALIDATION:")

    if missing:
        print("Missing fields:", missing)
    else:
        print("All required fields are present.")

except Exception as e:
    print("\nJSON PARSING FAILED")
    print("Error:", e)