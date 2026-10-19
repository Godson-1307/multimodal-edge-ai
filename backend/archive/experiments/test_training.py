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

from training_dataset import CORDTrainingDataset


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


print("\n========== TINY QLORA TRAINING ==========")

# --------------------------------------------------
# 1. Load processor
# --------------------------------------------------

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(MODEL_ID)


# --------------------------------------------------
# 2. Load model in 4-bit
# --------------------------------------------------

print("Loading model...")

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

print("Model loaded!")


# --------------------------------------------------
# 3. Prepare model for QLoRA
# --------------------------------------------------

model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    lora_config
)

print("\n========== LORA ==========")
model.print_trainable_parameters()
print("===========================")


# --------------------------------------------------
# 4. Load only 2 CORD samples
# --------------------------------------------------

dataset = CORDTrainingDataset("train")

print("\nTraining samples:", 2)


# --------------------------------------------------
# 5. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


# --------------------------------------------------
# 6. Training loop
# --------------------------------------------------

model.train()

for step in range(2):

    print(f"\n---------- STEP {step + 1}/2 ----------")

    sample = dataset[step]

    image = sample["image"]
    target = sample["target"]

    target_json = json.dumps(
        target,
        ensure_ascii=False
    )

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

    # User conversation
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

    # Full conversation
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

    # --------------------------------------------------
    # Create text
    # --------------------------------------------------

    user_text = processor.apply_chat_template(
        user_conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    full_text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )


    # --------------------------------------------------
    # Process user portion
    # --------------------------------------------------

    user_inputs = processor(
        text=[user_text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )


    # --------------------------------------------------
    # Process complete conversation
    # --------------------------------------------------

    inputs = processor(
        text=[full_text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )


    # --------------------------------------------------
    # Move tensors to GPU
    # --------------------------------------------------

    inputs = {
        key: value.to(model.device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }


    # --------------------------------------------------
    # Create labels
    # --------------------------------------------------

    labels = inputs["input_ids"].clone()


    # Ignore padding
    if processor.tokenizer.pad_token_id is not None:

        labels[
            labels == processor.tokenizer.pad_token_id
        ] = -100


    # Find user portion length
    user_length = user_inputs["input_ids"].shape[1]


    # Ignore user tokens
    labels[:, :user_length] = -100


    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    outputs = model(
        **inputs,
        labels=labels
    )

    loss = outputs.loss

    print("Loss:", loss.item())


    # --------------------------------------------------
    # Backpropagation
    # --------------------------------------------------

    optimizer.zero_grad()

    loss.backward()


    # --------------------------------------------------
    # Update LoRA weights
    # --------------------------------------------------

    optimizer.step()

    print("LoRA weights updated successfully.")


print("\n========== TRAINING SUCCESS ==========")
print("2 samples completed.")
print("Forward pass: SUCCESS")
print("Backward pass: SUCCESS")
print("Optimizer update: SUCCESS")
print("======================================")