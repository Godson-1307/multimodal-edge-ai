import torch
from transformers import BitsAndBytesConfig

print("=" * 50)
print("PHASE 8.3 - QUANTIZATION CONFIGURATION")
print("=" * 50)

# Current QLoRA quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

print("\nQuantization enabled:")
print(bnb_config.load_in_4bit)

print("\nQuantization type:")
print(bnb_config.bnb_4bit_quant_type)

print("\nCompute dtype:")
print(bnb_config.bnb_4bit_compute_dtype)

print("\nDouble quantization:")
print(bnb_config.bnb_4bit_use_double_quant)

print("\nCUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

print("\n" + "=" * 50)
print("CONFIGURATION CHECK COMPLETE")
print("=" * 50)