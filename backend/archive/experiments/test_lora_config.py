from peft import LoraConfig


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


print("========== LoRA CONFIG ==========")

print("Rank (r):", lora_config.r)
print("Alpha:", lora_config.lora_alpha)
print("Target modules:", lora_config.target_modules)
print("Dropout:", lora_config.lora_dropout)
print("Bias:", lora_config.bias)
print("Task type:", lora_config.task_type)

print("=================================")