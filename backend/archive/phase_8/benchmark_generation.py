import os
import json
import time
import torch

from PIL import Image
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
ADAPTER_DIR = "outputs/qlora_mini"

MAX_IMAGE_SIZE = 768
NUM_SAMPLES = 5

TOKEN_LIMITS = [64, 128, 256]


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

For line_items extract:
- name
- quantity
- price

Return ONLY a valid JSON object.
Do not add explanations.
Do not use markdown code blocks.
If a field is not visible, use null.
"""


# ============================================================
# IMAGE RESIZE
# ============================================================

def resize_image(image, max_size=768):

    width, height = image.size

    scale = min(
        max_size / width,
        max_size / height,
        1.0
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    if scale < 1.0:

        image = image.resize(
            (new_width, new_height)
        )

    return image


# ============================================================
# JSON VALIDATION
# ============================================================

def is_valid_json(text):

    try:

        json.loads(text)

        return True

    except Exception:

        return False


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("PHASE 8.5 — GENERATION OPTIMIZATION")
print("=" * 60)

print("\nLoading CORD dataset...")

dataset = load_dataset(
    "naver-clova-ix/cord-v2"
)

validation = dataset["validation"]

print(
    f"Validation samples available: {len(validation)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)


print("Loading 4-bit model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)


model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)


print("Loading QLoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_DIR
)

model.eval()

print("Model ready.")


# ============================================================
# GPU CHECK
# ============================================================

if torch.cuda.is_available():

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

else:

    print("\nWARNING: CUDA is not available.")


# ============================================================
# PREPARE CHAT TEMPLATE
# ============================================================

def prepare_inputs(image):

    image = resize_image(
        image,
        MAX_IMAGE_SIZE
    )

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
        return_tensors="pt"
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    return inputs


# ============================================================
# BENCHMARK FUNCTION
# ============================================================

def benchmark_token_limit(token_limit):

    print("\n" + "-" * 60)

    print(
        f"Testing max_new_tokens = {token_limit}"
    )

    print("-" * 60)

    times = []
    valid_json_count = 0
    total_generated_tokens = 0

    for i in range(NUM_SAMPLES):

        print(
            f"\nSample {i + 1}/{NUM_SAMPLES}"
        )

        image = validation[i]["image"]

        inputs = prepare_inputs(image)

        if torch.cuda.is_available():

            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():

            output_ids = model.generate(
                **inputs,
                max_new_tokens=token_limit,
                do_sample=False
            )

        if torch.cuda.is_available():

            torch.cuda.synchronize()

        end_time = time.perf_counter()

        elapsed = end_time - start_time

        times.append(elapsed)

        # ----------------------------------------------------
        # REMOVE INPUT TOKENS
        # ----------------------------------------------------

        input_length = inputs["input_ids"].shape[1]

        generated_ids = output_ids[
            0,
            input_length:
        ]

        generated_tokens = len(
            generated_ids
        )

        total_generated_tokens += generated_tokens

        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        response = processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        valid = is_valid_json(response)

        if valid:

            valid_json_count += 1

        print(
            f"Time: {elapsed:.2f} sec | "
            f"Tokens: {generated_tokens} | "
            f"JSON: {'YES' if valid else 'NO'}"
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    average_time = sum(times) / len(times)

    fastest = min(times)

    slowest = max(times)

    average_tokens = (
        total_generated_tokens
        / NUM_SAMPLES
    )

    json_accuracy = (
        valid_json_count
        / NUM_SAMPLES
    ) * 100

    print("\nRESULTS")

    print(
        f"Average inference time: "
        f"{average_time:.2f} sec"
    )

    print(
        f"Fastest inference: "
        f"{fastest:.2f} sec"
    )

    print(
        f"Slowest inference: "
        f"{slowest:.2f} sec"
    )

    print(
        f"Average generated tokens: "
        f"{average_tokens:.1f}"
    )

    print(
        f"Valid JSON: "
        f"{valid_json_count}/{NUM_SAMPLES} "
        f"({json_accuracy:.1f}%)"
    )

    return {
        "tokens": token_limit,
        "average_time": average_time,
        "fastest": fastest,
        "slowest": slowest,
        "average_generated_tokens": average_tokens,
        "valid_json": json_accuracy
    }


# ============================================================
# RUN ALL TOKEN LIMITS
# ============================================================

results = []

for token_limit in TOKEN_LIMITS:

    result = benchmark_token_limit(
        token_limit
    )

    results.append(result)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("FINAL GENERATION OPTIMIZATION COMPARISON")
print("=" * 80)

print(
    f"{'Max Tokens':<15}"
    f"{'Avg Time':<15}"
    f"{'Fastest':<15}"
    f"{'Slowest':<15}"
    f"{'Avg Tokens':<15}"
    f"{'Valid JSON':<15}"
)

print("-" * 80)

for result in results:

    print(
        f"{result['tokens']:<15}"
        f"{result['average_time']:.2f}s"
        f"{'':<10}"
        f"{result['fastest']:.2f}s"
        f"{'':<10}"
        f"{result['slowest']:.2f}s"
        f"{'':<10}"
        f"{result['average_generated_tokens']:.1f}"
        f"{'':<10}"
        f"{result['valid_json']:.1f}%"
    )


# ============================================================
# FIND FASTEST VALID CONFIGURATION
# ============================================================

valid_results = [
    r for r in results
    if r["valid_json"] == 100.0
]

if valid_results:

    best = min(
        valid_results,
        key=lambda x: x["average_time"]
    )

    print("\n")
    print("BEST CONFIGURATION")
    print("-" * 40)

    print(
        f"max_new_tokens: "
        f"{best['tokens']}"
    )

    print(
        f"Average inference time: "
        f"{best['average_time']:.2f} sec"
    )

    print(
        f"Average generated tokens: "
        f"{best['average_generated_tokens']:.1f}"
    )

    print(
        f"Valid JSON: "
        f"{best['valid_json']:.1f}%"
    )

else:

    print("\nNo configuration achieved 100% valid JSON.")


# ============================================================
# GPU MEMORY
# ============================================================

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

    print("\nGPU MEMORY")

    print(
        f"Allocated: {allocated:.2f} GB"
    )

    print(
        f"Reserved: {reserved:.2f} GB"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("PHASE 8.5 COMPLETE")
print("=" * 60)