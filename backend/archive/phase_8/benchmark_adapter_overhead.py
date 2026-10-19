import os
import json

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ADAPTER_DIR = "outputs/qlora_mini"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def get_directory_size(directory):
    total = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)

            if os.path.isfile(path):
                total += os.path.getsize(path)

    return total


def format_size(size_bytes):
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)

    if size_gb >= 1:
        return f"{size_gb:.2f} GB"

    return f"{size_mb:.2f} MB"


# --------------------------------------------------
# CHECK ADAPTER DIRECTORY
# --------------------------------------------------

print("=" * 60)
print("PHASE 8.4 — QLoRA ADAPTER OVERHEAD")
print("=" * 60)

if not os.path.exists(ADAPTER_DIR):
    print("\nERROR: Adapter directory not found!")
    print(f"Expected: {ADAPTER_DIR}")
    exit()

print(f"\nAdapter directory: {ADAPTER_DIR}")


# --------------------------------------------------
# ADAPTER FILE SIZES
# --------------------------------------------------

print("\n" + "-" * 60)
print("ADAPTER FILES")
print("-" * 60)

adapter_total_bytes = 0

for root, dirs, files in os.walk(ADAPTER_DIR):

    for file in files:

        path = os.path.join(root, file)

        if os.path.isfile(path):

            size = os.path.getsize(path)
            adapter_total_bytes += size

            print(f"{file:<35} {format_size(size)}")


print("\nTotal adapter directory size:")
print(format_size(adapter_total_bytes))


# --------------------------------------------------
# READ ADAPTER CONFIG
# --------------------------------------------------

config_path = os.path.join(
    ADAPTER_DIR,
    "adapter_config.json"
)

if os.path.exists(config_path):

    with open(config_path, "r") as f:
        config = json.load(f)

    print("\n" + "-" * 60)
    print("QLORA CONFIGURATION")
    print("-" * 60)

    print("LoRA rank:", config.get("r"))
    print("LoRA alpha:", config.get("lora_alpha"))
    print("LoRA dropout:", config.get("lora_dropout"))
    print("Target modules:", config.get("target_modules"))
    print("Task type:", config.get("task_type"))

else:

    print("\nWARNING: adapter_config.json not found.")


# --------------------------------------------------
# LOAD PYTORCH MODEL INFORMATION
# --------------------------------------------------

print("\n" + "-" * 60)
print("PARAMETER INFORMATION")
print("-" * 60)

try:

    import torch
    from transformers import AutoModelForMultimodalLM
    from peft import PeftModel

    print("\nLoading base model...")

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        device_map="auto"
    )

    print("Base model loaded.")

    print("\nLoading QLoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_DIR
    )

    print("QLoRA adapter loaded.")

    # --------------------------------------------------
    # PARAMETER COUNTS
    # --------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTotal parameters:")
    print(f"{total_params:,}")

    print("\nTrainable parameters:")
    print(f"{trainable_params:,}")

    trainable_percentage = (
        trainable_params / total_params
    ) * 100

    print("\nTrainable percentage:")
    print(f"{trainable_percentage:.4f}%")

    # --------------------------------------------------
    # ADAPTER PARAMETER COUNT
    # --------------------------------------------------

    adapter_params = 0

    for name, parameter in model.named_parameters():

        if "lora_" in name:
            adapter_params += parameter.numel()

    print("\nLoRA adapter parameters:")
    print(f"{adapter_params:,}")

    # --------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 3)
        )

        print("\nGPU:")
        print(torch.cuda.get_device_name(0))

        print("\nGPU memory allocated:")
        print(f"{allocated:.2f} GB")

        print("\nGPU memory reserved:")
        print(f"{reserved:.2f} GB")

except Exception as e:

    print("\nCould not load model for parameter analysis.")

    print("Error:")
    print(e)


# --------------------------------------------------
# FINAL ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("PHASE 8.4 RESULTS")
print("=" * 60)

print("\nAdapter size:")
print(format_size(adapter_total_bytes))

print("\nImportant observations:")

print("1. QLoRA trains only a small fraction of parameters.")

print("2. The LoRA adapter is much smaller than the base model.")

print("3. The base model remains the main storage component.")

print("4. The adapter can be stored separately from the base model.")

print("5. This makes QLoRA practical for edge deployment.")

print("\n" + "=" * 60)
print("PHASE 8.4 COMPLETE")
print("=" * 60)