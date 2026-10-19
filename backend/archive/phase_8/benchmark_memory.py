import os
import gc
import time
import torch

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig
)
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "outputs/qlora_mini"

NUM_SAMPLES = 5
MAX_IMAGE_SIZE = 768
MAX_NEW_TOKENS = 128

PROMPT = """
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

Return ONLY a valid JSON object.
Do not add explanations.
Do not use markdown code blocks.
If a field is not visible, use null.
"""


# ============================================================
# GPU MEMORY FUNCTION
# ============================================================

def gpu_memory():
    if not torch.cuda.is_available():
        return 0, 0, 0

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return allocated, reserved, peak


def print_memory(label):
    allocated, reserved, peak = gpu_memory()

    print(f"\n--- {label} ---")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Peak:     {peak:.2f} GB")


# ============================================================
# CLEAN GPU MEMORY
# ============================================================

def cleanup():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("PHASE 8.6 — GPU MEMORY OPTIMIZATION")
print("=" * 60)

print("\nGPU:")
print(torch.cuda.get_device_name(0))

cleanup()

print_memory("BEFORE MODEL LOADING")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)


print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
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

cleanup()

print_memory("AFTER MODEL LOADING")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading CORD validation dataset...")

dataset = load_dataset(
    "naver-clova-ix/cord-v2",
    split="validation"
)

dataset = dataset.select(
    range(min(NUM_SAMPLES, len(dataset)))
)

print(f"Samples selected: {len(dataset)}")


# ============================================================
# IMAGE RESIZE
# ============================================================

def resize_image(image):

    image = image.convert("RGB")

    width, height = image.size

    max_dimension = max(width, height)

    if max_dimension > MAX_IMAGE_SIZE:

        scale = MAX_IMAGE_SIZE / max_dimension

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize(
            (new_width, new_height)
        )

    return image


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(image):

    image = resize_image(image)

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
                    "text": PROMPT
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
        key: value.to(model.device)
        if torch.is_tensor(value)
        else value
        for key, value in inputs.items()
    }

    return inputs


# ============================================================
# BENCHMARK FUNCTION
# ============================================================

def benchmark(mode_name, use_inference_mode, use_cache):

    print("\n")
    print("=" * 60)
    print(mode_name)
    print("=" * 60)

    cleanup()

    total_time = 0
    peak_memory = 0

    for i, sample in enumerate(dataset):

        print(f"\nSample {i + 1}/{len(dataset)}")

        image = sample["image"]

        inputs = prepare_input(image)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.time()

        if use_inference_mode:

            with torch.inference_mode():

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=use_cache
                )

        else:

            with torch.no_grad():

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=use_cache
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start

        allocated, reserved, peak = gpu_memory()

        total_time += elapsed

        peak_memory = max(
            peak_memory,
            peak
        )

        print(f"Time: {elapsed:.2f} sec")
        print(f"Peak VRAM: {peak:.2f} GB")

        del inputs
        del output_ids

        cleanup()

    average_time = total_time / len(dataset)

    print("\nRESULT:")
    print(f"Average inference time: {average_time:.2f} sec")
    print(f"Peak GPU memory: {peak_memory:.2f} GB")

    return average_time, peak_memory


# ============================================================
# TEST 1
# ============================================================

normal_no_grad = benchmark(
    "TEST 1 — no_grad + use_cache=True",
    use_inference_mode=False,
    use_cache=True
)


# ============================================================
# TEST 2
# ============================================================

inference_mode = benchmark(
    "TEST 2 — inference_mode + use_cache=True",
    use_inference_mode=True,
    use_cache=True
)


# ============================================================
# TEST 3
# ============================================================

no_cache = benchmark(
    "TEST 3 — inference_mode + use_cache=False",
    use_inference_mode=True,
    use_cache=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("FINAL GPU MEMORY OPTIMIZATION COMPARISON")
print("=" * 80)

print(
    f"{'Configuration':<40}"
    f"{'Avg Time':<15}"
    f"{'Peak VRAM':<15}"
)

print("-" * 80)

print(
    f"{'no_grad + cache=True':<40}"
    f"{normal_no_grad[0]:.2f}s{'':<10}"
    f"{normal_no_grad[1]:.2f} GB"
)

print(
    f"{'inference_mode + cache=True':<40}"
    f"{inference_mode[0]:.2f}s{'':<10}"
    f"{inference_mode[1]:.2f} GB"
)

print(
    f"{'inference_mode + cache=False':<40}"
    f"{no_cache[0]:.2f}s{'':<10}"
    f"{no_cache[1]:.2f} GB"
)


# ============================================================
# DECISION
# ============================================================

print("\n")
print("=" * 60)
print("PHASE 8.6 DECISION")
print("=" * 60)

best_time = min(
    normal_no_grad[0],
    inference_mode[0],
    no_cache[0]
)

if inference_mode[0] <= normal_no_grad[0] * 1.05:

    print("\nKEEP: torch.inference_mode()")

    print(
        "Inference mode provides an efficient inference context "
        "without gradient tracking."
    )

else:

    print("\nKEEP: torch.no_grad()")

    print(
        "Inference mode did not provide a meaningful speed advantage "
        "in this benchmark."
    )


if no_cache[0] > inference_mode[0]:

    print("\nREJECT: use_cache=False")

    print(
        "Disabling the generation cache increased inference time."
    )

else:

    print("\nKEEP: use_cache=False")

    print(
        "Disabling cache did not hurt inference speed in this run."
    )


print("\nPHASE 8.6 COMPLETE")