import time
import torch
import json
import re

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
MAX_NEW_TOKENS = 256


# ============================================================
# GPU MEMORY FUNCTION
# ============================================================

def gpu_memory():

    if not torch.cuda.is_available():
        return 0.0

    return torch.cuda.max_memory_allocated() / (1024 ** 3)


# ============================================================
# IMAGE RESIZE
# ============================================================

def resize_image(image):

    image = image.convert("RGB")

    width, height = image.size

    scale = MAX_IMAGE_SIZE / max(width, height)

    if scale < 1:

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize(
            (new_width, new_height)
        )

    return image


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(text):

    try:
        return json.loads(text)

    except Exception:

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if match:

            try:
                return json.loads(match.group())

            except Exception:
                return None

    return None


# ============================================================
# QUANTIZATION CONFIGURATION
# ============================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)


# ============================================================
# RESET GPU MEMORY STATS
# ============================================================

if torch.cuda.is_available():

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("=" * 60)
print("PHASE 8.3.2 - 4-BIT QUANTIZATION BENCHMARK")
print("=" * 60)

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)


# ============================================================
# LOAD QUANTIZED MODEL
# ============================================================

print("\nLoading 4-bit quantized model...")

load_start = time.time()

model = AutoModelForMultimodalLM.from_pretrained(

    MODEL_ID,

    quantization_config=bnb_config,

    device_map="auto"
)

load_time = time.time() - load_start


print(f"Model loading time: {load_time:.2f} seconds")


# ============================================================
# LOAD QLORA ADAPTER
# ============================================================

print("\nLoading QLoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()


# ============================================================
# GPU INFORMATION
# ============================================================

if torch.cuda.is_available():

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

    print(
        f"Memory after loading: "
        f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB"
    )


# ============================================================
# LOAD CORD
# ============================================================

print("\nLoading CORD validation dataset...")

dataset = load_dataset(
    "naver-clova-ix/cord-v2"
)

validation = dataset["validation"]


# ============================================================
# PROMPT
# ============================================================

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


# ============================================================
# RUN BENCHMARK
# ============================================================

times = []

json_valid = 0

print("\n" + "=" * 60)
print("RUNNING INFERENCE")
print("=" * 60)


for i in range(NUM_SAMPLES):

    print(f"\nSample {i + 1}/{NUM_SAMPLES}")

    image = validation[i]["image"]

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
        key: value.to(model.device)
        if torch.is_tensor(value)
        else value
        for key, value in inputs.items()
    }

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start

    times.append(elapsed)

    # Only decode newly generated tokens
    input_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[:, input_length:]

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    parsed = parse_json_response(response)

    if parsed is not None:
        json_valid += 1

    print(
        f"Time: {elapsed:.2f} sec | "
        f"JSON: {'YES' if parsed is not None else 'NO'}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

average_time = sum(times) / len(times)

fastest = min(times)

slowest = max(times)

peak_vram = gpu_memory()


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("4-BIT QUANTIZATION RESULTS")
print("=" * 60)

print(f"Samples tested: {NUM_SAMPLES}")

print(
    f"Average inference time: "
    f"{average_time:.2f} seconds"
)

print(
    f"Fastest inference: "
    f"{fastest:.2f} seconds"
)

print(
    f"Slowest inference: "
    f"{slowest:.2f} seconds"
)

print(
    f"Model loading time: "
    f"{load_time:.2f} seconds"
)

print(
    f"Peak GPU memory: "
    f"{peak_vram:.2f} GB"
)

print(
    f"Valid JSON: "
    f"{json_valid}/{NUM_SAMPLES} "
    f"({json_valid / NUM_SAMPLES * 100:.1f}%)"
)

print("\nQuantization configuration:")

print("4-bit: True")
print("Type: NF4")
print("Compute: BF16")
print("Double Quantization: True")

print("\n" + "=" * 60)
print("PHASE 8.3.2 COMPLETE")
print("=" * 60)