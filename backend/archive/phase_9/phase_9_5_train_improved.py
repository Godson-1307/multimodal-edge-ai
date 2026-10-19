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

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


# ============================================================
# PHASE 9.5 - IMPROVED QLORA TRAINING
# ============================================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

TRAIN_FILE = "data/processed/train_improved.jsonl"

OUTPUT_DIR = "outputs/qlora_improved"

NUM_SAMPLES = 100

MAX_IMAGE_SIZE = 768

LEARNING_RATE = 1e-4


# ============================================================
# IMPROVED TRAINING PROMPT
# ============================================================

PROMPT = """
You are an invoice information extraction system.

Carefully inspect the ENTIRE invoice image and extract all visible
information into the JSON structure below.

IMPORTANT RULES:

1. Extract EVERY visible line item. Do not skip any item.
2. Preserve each product/item name as accurately as possible.
3. For every line item, identify its quantity and price from the
   SAME row whenever they are visible.
4. Do not invent information.
5. If a field is genuinely not visible, return null.
6. Carefully distinguish between:
   - subtotal
   - tax
   - total amount / grand total
7. For total_amount, use the final payable/grand total shown on
   the invoice, not the subtotal.
8. For total_tax, use the tax amount shown on the invoice.
9. Read numbers carefully, including commas and decimal points.
10. Return ONLY one valid JSON object.
11. Do NOT include explanations.
12. Do NOT use markdown.
13. Before returning the answer, check that all visible line items
    have been included.

Return exactly this structure:

{
  "total_amount": null,
  "total_tax": null,
  "line_items": [
    {
      "name": null,
      "quantity": null,
      "price": null
    }
  ]
}
"""


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("PHASE 9.5 - IMPROVED QLORA TRAINING")
print("=" * 70)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available.")

print("\nGPU:")
print(torch.cuda.get_device_name(0))

print("\nCUDA:")
print(torch.version.cuda)

torch.cuda.empty_cache()

print("\nInitial GPU memory:")
print(
    f"Allocated: "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)

print(
    f"Reserved:  "
    f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
)


# ============================================================
# LOAD IMPROVED TARGET DATA
# ============================================================

print("\n" + "=" * 70)
print("Loading improved targets...")
print("=" * 70)

target_dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
    split="train"
)

print(
    f"Improved target samples: "
    f"{len(target_dataset)}"
)


# ============================================================
# LOAD ORIGINAL CORD DATASET
# ============================================================

print("\n" + "=" * 70)
print("Loading original CORD images...")
print("=" * 70)

cord_dataset = load_dataset(
    "naver-clova-ix/cord-v2",
    split="train"
)

print(
    f"CORD training samples: "
    f"{len(cord_dataset)}"
)


# ============================================================
# LIMIT TRAINING SAMPLES
# ============================================================

num_train = min(
    NUM_SAMPLES,
    len(target_dataset),
    len(cord_dataset)
)

target_dataset = target_dataset.select(
    range(num_train)
)

print(
    f"\nSamples selected for training: "
    f"{num_train}"
)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\n" + "=" * 70)
print("Loading processor...")
print("=" * 70)

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)


# ============================================================
# 4-BIT QUANTIZATION CONFIG
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("Loading Qwen2-VL model...")
print("=" * 70)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

print("\nModel loaded.")

print(
    f"GPU allocated after model loading: "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)

print(
    f"GPU reserved after model loading:  "
    f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
)


# ============================================================
# PREPARE MODEL FOR QLORA
# ============================================================

print("\n" + "=" * 70)
print("Preparing model for QLoRA...")
print("=" * 70)

model = prepare_model_for_kbit_training(model)


# ============================================================
# LORA CONFIG
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# ============================================================
# ATTACH LORA
# ============================================================

model = get_peft_model(
    model,
    lora_config
)

print("\nLoRA configuration:")

model.print_trainable_parameters()


# ============================================================
# OPTIMIZER
# ============================================================

trainable_parameters = [
    p for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

start_time = time.time()

loss_values = []

model.train()

for step, sample in enumerate(
    target_dataset,
    start=1
):

    print(
        f"\nProcessing sample "
        f"{step}/{num_train}"
    )

    # --------------------------------------------------------
    # GET IMAGE ID
    # --------------------------------------------------------

    image_id = sample["image_id"]

    print(
        f"Image ID: {image_id}"
    )

    # --------------------------------------------------------
    # GET ORIGINAL CORD IMAGE
    # --------------------------------------------------------

    image = cord_dataset[image_id]["image"]

    if image is None:
        raise ValueError(
            f"Image not found for image_id={image_id}"
        )

    # Make sure RGB
    image = image.convert("RGB")

    # --------------------------------------------------------
    # RESIZE IMAGE
    # --------------------------------------------------------

    image.thumbnail(
        (
            MAX_IMAGE_SIZE,
            MAX_IMAGE_SIZE
        )
    )

    print(
        f"Image size: {image.size}"
    )

    # --------------------------------------------------------
    # GET IMPROVED TARGET
    # --------------------------------------------------------

    target = sample["target"]

    target_text = json.dumps(
        target,
        ensure_ascii=False
    )

    # --------------------------------------------------------
    # BUILD CHAT MESSAGES
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": PROMPT
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": target_text
                }
            ]
        }
    ]

    # --------------------------------------------------------
    # APPLY CHAT TEMPLATE
    # --------------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # --------------------------------------------------------
    # PROCESS IMAGE + TEXT
    # --------------------------------------------------------

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )

    # --------------------------------------------------------
    # MOVE TENSORS TO GPU
    # --------------------------------------------------------

    processed_inputs = {}

    for key, value in inputs.items():

        if hasattr(value, "to"):
            processed_inputs[key] = value.to(
                model.device
            )
        else:
            processed_inputs[key] = value

    inputs = processed_inputs

    # --------------------------------------------------------
    # CREATE LABELS
    # --------------------------------------------------------

    labels = inputs["input_ids"].clone()

    # --------------------------------------------------------
    # FIND ASSISTANT START
    # --------------------------------------------------------

    assistant_marker = (
        "<|im_start|>assistant\n"
    )

    marker_ids = processor.tokenizer(
        assistant_marker,
        add_special_tokens=False
    )["input_ids"]

    input_ids = (
        inputs["input_ids"][0]
        .tolist()
    )

    assistant_start = None

    for i in range(
        len(input_ids) - len(marker_ids) + 1
    ):

        if input_ids[
            i:i + len(marker_ids)
        ] == marker_ids:

            assistant_start = (
                i + len(marker_ids)
            )

            break

    if assistant_start is None:

        raise RuntimeError(
            "Could not find assistant marker "
            "in tokenized input."
        )

    # --------------------------------------------------------
    # MASK USER TOKENS
    # --------------------------------------------------------

    labels[:, :assistant_start] = -100

    inputs["labels"] = labels

    # --------------------------------------------------------
    # ZERO GRADIENT
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):

        outputs = model(
            **inputs
        )

    loss = outputs.loss

    # --------------------------------------------------------
    # BACKWARD PASS
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # GRADIENT CLIPPING
    # --------------------------------------------------------

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # --------------------------------------------------------
    # OPTIMIZER STEP
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # STORE LOSS
    # --------------------------------------------------------

    loss_value = loss.item()

    loss_values.append(
        loss_value
    )

    # --------------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------------

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print(
        f"STEP {step}/{num_train} "
        f"| Loss: {loss_value:.6f} "
        f"| GPU: {allocated:.2f} GB "
        f"| Reserved: {reserved:.2f} GB"
    )


# ============================================================
# TRAINING COMPLETE
# ============================================================

training_time = (
    time.time() - start_time
)

average_loss = (
    sum(loss_values)
    / len(loss_values)
)


# ============================================================
# SAVE ADAPTER
# ============================================================

print("\n" + "=" * 70)
print("SAVING IMPROVED QLORA ADAPTER")
print("=" * 70)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

model.save_pretrained(
    OUTPUT_DIR
)

processor.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# FINAL GPU MEMORY
# ============================================================

peak_memory = (
    torch.cuda.max_memory_allocated()
    / 1024**3
)

final_allocated = (
    torch.cuda.memory_allocated()
    / 1024**3
)

final_reserved = (
    torch.cuda.memory_reserved()
    / 1024**3
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 9.5 COMPLETE")
print("=" * 70)

print(
    f"Samples trained:       {num_train}"
)

print(
    f"Average training loss: "
    f"{average_loss:.6f}"
)

print(
    f"Training time:         "
    f"{training_time:.2f} sec"
)

print(
    f"Peak GPU memory:       "
    f"{peak_memory:.2f} GB"
)

print(
    f"Final GPU allocated:   "
    f"{final_allocated:.2f} GB"
)

print(
    f"Final GPU reserved:    "
    f"{final_reserved:.2f} GB"
)

print(
    f"Adapter saved to:      "
    f"{OUTPUT_DIR}"
)

print("\nIMPROVED QLORA TRAINING SUCCESS")
print("=" * 70)