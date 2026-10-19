# ============================================
# FINAL EDGE CONFIGURATION
# Multi-Modal Edge AI
# ============================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "outputs/qlora_mini"

# Quantization
LOAD_IN_4BIT = True
QUANT_TYPE = "nf4"
COMPUTE_DTYPE = "bfloat16"
USE_DOUBLE_QUANT = True

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Image configuration
MAX_IMAGE_WIDTH = 384
MAX_IMAGE_HEIGHT = 576

# Generation
MAX_NEW_TOKENS = 128
DO_SAMPLE = False
USE_CACHE = True

# Inference
BATCH_SIZE = 1
USE_INFERENCE_MODE = True

print("============================================")
print("FINAL EDGE CONFIGURATION")
print("============================================")

print("Model:", MODEL_ID)
print("Adapter:", ADAPTER_PATH)

print("\nQuantization:")
print("4-bit:", LOAD_IN_4BIT)
print("Type:", QUANT_TYPE)
print("Compute dtype:", COMPUTE_DTYPE)
print("Double quant:", USE_DOUBLE_QUANT)

print("\nQLoRA:")
print("Rank:", LORA_R)
print("Alpha:", LORA_ALPHA)
print("Dropout:", LORA_DROPOUT)
print("Targets:", LORA_TARGET_MODULES)

print("\nImage:")
print("Maximum size:", f"{MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}")

print("\nGeneration:")
print("Max new tokens:", MAX_NEW_TOKENS)
print("Sampling:", DO_SAMPLE)
print("Use cache:", USE_CACHE)

print("\nInference:")
print("Batch size:", BATCH_SIZE)
print("Inference mode:", USE_INFERENCE_MODE)

print("\n============================================")
print("CONFIGURATION READY")
print("============================================")