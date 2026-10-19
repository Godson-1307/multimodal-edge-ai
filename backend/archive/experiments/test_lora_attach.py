import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


# -----------------------------
# 1. Load processor
# -----------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)


# -----------------------------
# 2. 4-bit quantization
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)


# -----------------------------
# 3. Load model
# -----------------------------
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)


print("\n========== MODEL LOADED ==========")
print("Model loaded successfully!")
print("==================================\n")


# -----------------------------
# 4. LoRA configuration
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# -----------------------------
# 5. Attach LoRA
# -----------------------------
model = get_peft_model(model, lora_config)


print("========== LoRA ATTACHED ==========")

model.print_trainable_parameters()

print("===================================")