import torch
import json

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


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


print("\n========== QLORA TRAINING ==========")
print("Loading processor...")


# ============================================================
# PROCESSOR
# ============================================================

processor = AutoProcessor.from_pretrained(MODEL_ID)


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
# LOAD MODEL
# ============================================================

print("Loading model...")


model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)


print("Model loaded successfully!")


# ============================================================
# PREPARE MODEL FOR QLORA TRAINING
# ============================================================

model = prepare_model_for_kbit_training(model)


# ============================================================
# LORA CONFIGURATION
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# Attach LoRA adapters
model = get_peft_model(
    model,
    lora_config
)


print("\n========== LORA ==========")

model.print_trainable_parameters()

print("==========================")


# ============================================================
# GPU CHECK
# ============================================================

print("\n========== GPU ==========")


if torch.cuda.is_available():

    print("CUDA available: True")
    print("GPU:", torch.cuda.get_device_name(0))

else:

    print("CUDA available: False")


print("==========================")


# ============================================================
# LOAD CORD DATASET
# ============================================================

from training_dataset import CORDTrainingDataset


print("\n========== CORD DATASET ==========")


train_dataset = CORDTrainingDataset("train")


print("Training samples:", len(train_dataset))


sample = train_dataset[0]


print("Image size:", sample["image"].size)
print("Target keys:", sample["target"].keys())


print("==================================")


# ============================================================
# CREATE TRAINING EXAMPLE
# ============================================================

print("\n========== TRAINING EXAMPLE ==========")


sample = train_dataset[0]


image = sample["image"]
target = sample["target"]


# Convert target dictionary to JSON
target_json = json.dumps(
    target,
    ensure_ascii=False
)


# ============================================================
# TRAINING PROMPT
# ============================================================

prompt = """
Extract the following information from this document:

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
If a field is not visible, use null.
"""


# ============================================================
# CREATE CONVERSATION
# ============================================================

conversation = [
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
    },

    {
        "role": "assistant",

        "content": [
            {
                "type": "text",
                "text": target_json
            }
        ]
    }
]


print("Image:", image.size)

print("Target JSON:")
print(target_json)


print("\nConversation created successfully!")

print("======================================")


# ============================================================
# PROCESSOR TEST
# ============================================================

print("\n========== PROCESSOR ==========")


text = processor.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=False
)


inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt"
)


print("Processor keys:")
print(inputs.keys())


print("\nInput shapes:")


for key, value in inputs.items():

    if hasattr(value, "shape"):

        print(key, value.shape)


print("\nProcessor conversion successful!")

print("================================")


# ============================================================
# LABEL BOUNDARY TEST
# ============================================================

print("\n========== LABEL BOUNDARY TEST ==========")


# ------------------------------------------------------------
# User-only conversation
# ------------------------------------------------------------

user_conversation = [
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


# ------------------------------------------------------------
# User portion
# ------------------------------------------------------------

user_text = processor.apply_chat_template(
    user_conversation,
    tokenize=False,
    add_generation_prompt=True
)


# ------------------------------------------------------------
# Full conversation
# ------------------------------------------------------------

full_text = processor.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=False
)


# ------------------------------------------------------------
# Tokenize user portion
# ------------------------------------------------------------

user_inputs = processor(
    text=[user_text],
    images=[image],
    padding=True,
    return_tensors="pt"
)


# ------------------------------------------------------------
# Tokenize full conversation
# ------------------------------------------------------------

full_inputs = processor(
    text=[full_text],
    images=[image],
    padding=True,
    return_tensors="pt"
)


# ------------------------------------------------------------
# Calculate token lengths
# ------------------------------------------------------------

user_length = user_inputs["input_ids"].shape[1]

full_length = full_inputs["input_ids"].shape[1]


print("User tokens:", user_length)

print(
    "Full conversation tokens:",
    full_length
)

print(
    "Assistant tokens:",
    full_length - user_length
)


print("\nLabel boundary test successful!")

print("==========================================")


# ============================================================
# TRAINING LABEL TEST
# ============================================================

print("\n========== TRAINING LABEL TEST ==========")


# ------------------------------------------------------------
# Process full conversation
# ------------------------------------------------------------

inputs = processor(
    text=[full_text],
    images=[image],
    padding=True,
    return_tensors="pt"
)


# ------------------------------------------------------------
# Move tensors to GPU
# ------------------------------------------------------------

inputs = {
    key: value.to(model.device)
    if hasattr(value, "to")
    else value

    for key, value in inputs.items()
}


# ------------------------------------------------------------
# Create labels
# ------------------------------------------------------------

labels = inputs["input_ids"].clone()


# ------------------------------------------------------------
# Ignore padding tokens
# ------------------------------------------------------------

if processor.tokenizer.pad_token_id is not None:

    labels[
        labels == processor.tokenizer.pad_token_id
    ] = -100


# ------------------------------------------------------------
# Ignore user/image portion
# ------------------------------------------------------------

labels[:, :user_length] = -100


# ------------------------------------------------------------
# Display label information
# ------------------------------------------------------------

print(
    "Input IDs:",
    inputs["input_ids"].shape
)

print(
    "Labels:",
    labels.shape
)


print(
    "Ignored tokens:",
    (labels == -100).sum().item()
)


print(
    "Training tokens:",
    (labels != -100).sum().item()
)


print("\n========== LABEL TEST SUCCESS ==========")

print("User portion masked.")

print("Assistant portion selected for training.")

print("========================================")

print("\n========== MASKED FORWARD PASS ==========")

with torch.no_grad():
    outputs = model(
        **inputs,
        labels=labels
    )

print("Masked Loss:", outputs.loss.item())

print("\n========== MASKED FORWARD PASS SUCCESS ==========")
print("Model calculated loss only on training tokens.")
print("===============================================")