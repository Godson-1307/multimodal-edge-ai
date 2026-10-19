import torch
from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


# 4-bit QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


print("Loading Qwen2-VL...")

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

print("\nModel loaded successfully!")
print("\n========== POSSIBLE LoRA TARGET LAYERS ==========")

target_keywords = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

count = 0

for name, module in model.named_modules():

    if any(keyword in name for keyword in target_keywords):

        print(f"{name}  ->  {module.__class__.__name__}")

        count += 1

print("\n==============================================")
print("Total matching layers:", count)
print("==============================================")