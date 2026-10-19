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

NUM_SAMPLES = 10
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
# GPU MEMORY
# ============================================================

def get_memory():

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return allocated, reserved, peak


def print_memory(label):

    allocated, reserved, peak = get_memory()

    print(f"\n--- {label} ---")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Peak:      {peak:.2f} GB")


# ============================================================
# CLEAN MEMORY
# ============================================================

def cleanup():

    gc.collect()

    torch.cuda.empty_cache()


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("PHASE 8.8 — STRESS TEST")
print("=" * 70)

print("\nGPU:")
print(torch.cuda.get_device_name(0))

cleanup()

torch.cuda.reset_peak_memory_stats()

print_memory("BEFORE MODEL LOADING")


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)


# ============================================================
# LOAD MODEL
# ============================================================

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

torch.cuda.reset_peak_memory_stats()

print_memory("AFTER MODEL LOADING")


# ============================================================
# LOAD CORD
# ============================================================

print("\nLoading CORD validation dataset...")

dataset = load_dataset(

    "naver-clova-ix/cord-v2",

    split="validation"
)


# Use first 10 samples
dataset = dataset.select(
    range(min(NUM_SAMPLES, len(dataset)))
)

print(
    f"Samples selected: {len(dataset)}"
)


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
# STRESS TEST
# ============================================================

print("\n")
print("=" * 70)
print("STARTING STRESS TEST")
print("=" * 70)

times = []

memory_values = []

successful = 0

failed = 0


for i, sample in enumerate(dataset):

    print("\n" + "-" * 70)

    print(
        f"PROCESSING INVOICE {i + 1}/{len(dataset)}"
    )

    try:

        image = sample["image"]

        inputs = prepare_input(image)

        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        # ----------------------------------------------------
        # OPTIMIZED INFERENCE
        # ----------------------------------------------------

        with torch.inference_mode():

            output_ids = model.generate(

                **inputs,

                max_new_tokens=MAX_NEW_TOKENS,

                do_sample=False,

                use_cache=True
            )

        torch.cuda.synchronize()

        elapsed = time.time() - start_time

        allocated, reserved, peak = get_memory()

        times.append(elapsed)

        memory_values.append(allocated)

        successful += 1

        print(
            f"Inference time: {elapsed:.2f} sec"
        )

        print(
            f"Allocated VRAM: {allocated:.2f} GB"
        )

        print(
            f"Reserved VRAM:  {reserved:.2f} GB"
        )

        print(
            f"Peak VRAM:      {peak:.2f} GB"
        )

        # ----------------------------------------------------
        # IMPORTANT MEMORY CLEANUP
        # ----------------------------------------------------

        del inputs

        del output_ids

        cleanup()

    except Exception as e:

        failed += 1

        print(
            f"ERROR: {e}"
        )

        cleanup()


# ============================================================
# FINAL MEMORY
# ============================================================

torch.cuda.synchronize()

final_allocated, final_reserved, final_peak = get_memory()


# ============================================================
# CALCULATE RESULTS
# ============================================================

if times:

    average_time = sum(times) / len(times)

    fastest = min(times)

    slowest = max(times)

else:

    average_time = 0

    fastest = 0

    slowest = 0


# ============================================================
# STRESS TEST RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("FINAL STRESS TEST RESULTS")
print("=" * 80)

print(
    f"Total samples:       {len(dataset)}"
)

print(
    f"Successful:           {successful}"
)

print(
    f"Failed:               {failed}"
)

print(
    f"Average time:         {average_time:.2f} sec"
)

print(
    f"Fastest inference:    {fastest:.2f} sec"
)

print(
    f"Slowest inference:    {slowest:.2f} sec"
)

print(
    f"Final allocated VRAM:  {final_allocated:.2f} GB"
)

print(
    f"Final reserved VRAM:   {final_reserved:.2f} GB"
)

print(
    f"Overall peak VRAM:     {final_peak:.2f} GB"
)


# ============================================================
# MEMORY STABILITY
# ============================================================

print("\n")
print("=" * 80)
print("MEMORY STABILITY CHECK")
print("=" * 80)


if memory_values:

    first_memory = memory_values[0]

    last_memory = memory_values[-1]

    difference = last_memory - first_memory

    print(
        f"First sample allocated: {first_memory:.2f} GB"
    )

    print(
        f"Last sample allocated:  {last_memory:.2f} GB"
    )

    print(
        f"Difference:             {difference:+.2f} GB"
    )

    if difference <= 0.10:

        print(
            "\nMEMORY STABLE"
        )

        print(
            "No significant GPU memory growth detected."
        )

    else:

        print(
            "\nMEMORY GROWTH DETECTED"
        )

        print(
            "Further memory investigation may be required."
        )


# ============================================================
# INFERENCE STABILITY
# ============================================================

print("\n")
print("=" * 80)
print("INFERENCE STABILITY")
print("=" * 80)


if failed == 0:

    print(
        "ALL SAMPLES PROCESSED SUCCESSFULLY"
    )

else:

    print(
        f"{failed} sample(s) failed."
    )


# ============================================================
# FINAL DECISION
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 8.8 DECISION")
print("=" * 80)


if failed == 0 and memory_values:

    if difference <= 0.10:

        print(
            "\nKEEP CURRENT CONFIGURATION"
        )

        print(
            "The optimized inference pipeline remained "
            "stable across repeated invoice processing."
        )

    else:

        print(
            "\nINVESTIGATE MEMORY GROWTH"
        )

else:

    print(
        "\nSTRESS TEST FAILED"
    )


print("\nPHASE 8.8 COMPLETE")