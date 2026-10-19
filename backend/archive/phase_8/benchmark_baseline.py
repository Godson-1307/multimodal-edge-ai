import time
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig
)
from peft import PeftModel


# ============================================
# CONFIGURATION
# ============================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "outputs/qlora_mini"

NUM_SAMPLES = 5
MAX_IMAGE_SIZE = 768


# ============================================
# GPU MEMORY FUNCTION
# ============================================

def gpu_memory():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    return allocated, reserved


# ============================================
# GPU CHECK
# ============================================

print("\n============================================")
print("PHASE 8.1 - PERFORMANCE BASELINE")
print("============================================")

if not torch.cuda.is_available():
    print("CUDA is not available.")
    exit()

print("\nGPU:")
print(torch.cuda.get_device_name(0))

print("\nGPU Memory Before Model Loading:")
allocated, reserved = gpu_memory()
print("Allocated:", round(allocated, 2), "GB")
print("Reserved:", round(reserved, 2), "GB")


# ============================================
# LOAD MODEL
# ============================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

print("Loading base model...")

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Loading QLoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

print("Model loaded successfully.")


# ============================================
# MEMORY AFTER MODEL
# ============================================

print("\nGPU Memory After Model Loading:")

allocated, reserved = gpu_memory()

print("Allocated:", round(allocated, 2), "GB")
print("Reserved:", round(reserved, 2), "GB")


# ============================================
# LOAD VALIDATION DATA
# ============================================

print("\nLoading CORD validation dataset...")

dataset = load_dataset(
    "naver-clova-ix/cord-v2",
    split="validation"
)

dataset = dataset.select(range(NUM_SAMPLES))

print("Samples:", len(dataset))


# ============================================
# PROMPT
# ============================================

prompt = """
Extract the following information from this invoice:

- vendor_name
- invoice_number
- invoice_date
- gstin
- state
- state_code
- total_tax
- total_amount
- line_items

For line_items extract:
- name
- quantity
- price

Return ONLY a valid JSON object.
Do not add explanations.
Do not use markdown code blocks.
If a field is not visible, use null.
"""


# ============================================
# BENCHMARK
# ============================================

times = []

torch.cuda.reset_peak_memory_stats()

print("\n============================================")
print("STARTING BENCHMARK")
print("============================================")

for i, sample in enumerate(dataset):

    print("\nSample", i + 1, "/", NUM_SAMPLES)

    image = sample["image"]

    # Resize image
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

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
        padding=True,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    # GPU synchronization
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    torch.cuda.synchronize()

    end_time = time.perf_counter()

    inference_time = end_time - start_time

    times.append(inference_time)

    print("Original image:", sample["image"].size)
    print("Processed image:", image.size)
    print("Inference time:", round(inference_time, 2), "seconds")


# ============================================
# FINAL MEMORY
# ============================================

allocated, reserved = gpu_memory()

peak_memory = (
    torch.cuda.max_memory_allocated()
    / (1024 ** 3)
)


# ============================================
# RESULTS
# ============================================

average_time = sum(times) / len(times)

print("\n============================================")
print("BASELINE RESULTS")
print("============================================")

print("Samples tested:", NUM_SAMPLES)

print(
    "Average inference time:",
    round(average_time, 2),
    "seconds"
)

print(
    "Fastest inference:",
    round(min(times), 2),
    "seconds"
)

print(
    "Slowest inference:",
    round(max(times), 2),
    "seconds"
)

print(
    "Peak GPU memory:",
    round(peak_memory, 2),
    "GB"
)

print(
    "Final GPU allocated:",
    round(allocated, 2),
    "GB"
)

print(
    "Final GPU reserved:",
    round(reserved, 2),
    "GB"
)

print("\n============================================")
print("PHASE 8.1 SUCCESS")
print("============================================")