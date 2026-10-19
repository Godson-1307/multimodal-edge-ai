import json
import time
import torch

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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

NUM_SAMPLES = 20

OUTPUT_DIR = "outputs/qlora_mini"

LEARNING_RATE = 1e-4

MAX_IMAGE_SIZE = 768


# ============================================================
# GPU CHECK
# ============================================================

print("\n========== DEVICE CHECK ==========")

if torch.cuda.is_available():

    device = torch.device("cuda")

    print("CUDA available: YES")
    print("GPU:", torch.cuda.get_device_name(0))

    total_memory = torch.cuda.get_device_properties(0).total_memory

    print(
        "GPU Memory:",
        round(total_memory / (1024 ** 3), 2),
        "GB"
    )

else:

    device = torch.device("cpu")

    print("CUDA available: NO")
    print("WARNING: Training will run on CPU.")

print("==================================\n")


# ============================================================
# GPU MEMORY FUNCTION
# ============================================================

def print_gpu_memory(label):

    if torch.cuda.is_available():

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)

        reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(f"\n--- GPU MEMORY: {label} ---")

        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print(f"Peak:      {peak:.2f} GB")


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

print("Processor loaded successfully.")


# ============================================================
# 4-BIT QUANTIZATION CONFIG
# ============================================================

print("\nCreating 4-bit quantization configuration...")

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)

print("4-bit configuration ready.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen2-VL model...")

print_gpu_memory("Before model loading")

model = AutoModelForMultimodalLM.from_pretrained(

    MODEL_ID,

    quantization_config=bnb_config,

    device_map="auto"
)

print("\nModel loaded successfully.")

print_gpu_memory("After model loading")


# ============================================================
# PREPARE MODEL FOR K-BIT TRAINING
# ============================================================

print("\nPreparing model for QLoRA training...")

model = prepare_model_for_kbit_training(model)

model.config.use_cache = False

print("Model prepared successfully.")


# ============================================================
# LoRA CONFIGURATION
# ============================================================

print("\nCreating LoRA configuration...")

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
# ATTACH LoRA
# ============================================================

print("\nAttaching LoRA adapters...")

model = get_peft_model(
    model,
    lora_config
)

print("\n========== LoRA PARAMETERS ==========")

model.print_trainable_parameters()

print("=====================================")


# ============================================================
# LOAD CORD DATASET
# ============================================================

print("\nLoading CORD training dataset...")

dataset = CORDTrainingDataset("train")

print("Total CORD training samples:", len(dataset))

print("Samples selected for mini training:", NUM_SAMPLES)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# TRAINING MODE
# ============================================================

model.train()

print("\n========== TRAINING STARTED ==========")

start_time = time.time()

losses = []


# ============================================================
# TRAINING LOOP
# ============================================================

for step in range(NUM_SAMPLES):

    print(
        f"\n---------- STEP {step + 1}/{NUM_SAMPLES} ----------"
    )

    # --------------------------------------------------------
    # LOAD SAMPLE
    # --------------------------------------------------------

    sample = dataset[step]

    image = sample["image"]

    target = sample["target"]


    # --------------------------------------------------------
    # RESIZE IMAGE
    # --------------------------------------------------------

    width, height = image.size

    max_dimension = max(width, height)

    if max_dimension > MAX_IMAGE_SIZE:

        scale = MAX_IMAGE_SIZE / max_dimension

        new_width = int(width * scale)

        new_height = int(height * scale)

        image = image.resize(
            (new_width, new_height)
        )


    print(
        "Image size:",
        image.size
    )


    # --------------------------------------------------------
    # CONVERT TARGET TO JSON
    # --------------------------------------------------------

    target_json = json.dumps(
        target,
        ensure_ascii=False
    )


    # --------------------------------------------------------
    # CREATE CHAT MESSAGE
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

                    "text": (
                        "Extract the information from this "
                        "receipt image and return the result "
                        "as JSON.\n\n"
                        "Required fields:\n"
                        "vendor_name\n"
                        "invoice_number\n"
                        "invoice_date\n"
                        "gstin\n"
                        "state\n"
                        "state_code\n"
                        "total_tax\n"
                        "total_amount\n"
                        "line_items\n\n"
                        "Return only valid JSON."
                    )
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


    # --------------------------------------------------------
    # APPLY CHAT TEMPLATE
    # --------------------------------------------------------

    text = processor.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=False
    )


    # --------------------------------------------------------
    # PREPARE INPUTS
    # --------------------------------------------------------

    inputs = processor(

        text=[text],

        images=[image],

        return_tensors="pt"
    )


    # --------------------------------------------------------
    # MOVE INPUTS TO MODEL DEVICE
    # --------------------------------------------------------

    model_device = next(
        model.parameters()
    ).device

    inputs = {
        key: value.to(model_device)
        if torch.is_tensor(value)
        else value

        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # CREATE LABELS
    # --------------------------------------------------------

    labels = inputs["input_ids"].clone()


    # --------------------------------------------------------
    # FIND USER / ASSISTANT SPLIT
    # --------------------------------------------------------

    assistant_marker = processor.tokenizer.encode(

        "<|im_start|>assistant\n",

        add_special_tokens=False
    )


    assistant_start = None


    input_ids = inputs["input_ids"][0].tolist()


    marker_length = len(assistant_marker)


    for i in range(
        len(input_ids) - marker_length + 1
    ):

        if input_ids[
            i:i + marker_length
        ] == assistant_marker:

            assistant_start = i + marker_length

            break


    # --------------------------------------------------------
    # MASK USER TOKENS
    # --------------------------------------------------------

    if assistant_start is not None:

        labels[:, :assistant_start] = -100

    else:

        print(
            "WARNING: Assistant marker not found."
        )

        print(
            "Training on all tokens."
        )


    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    outputs = model(

        **inputs,

        labels=labels
    )


    loss = outputs.loss


    # --------------------------------------------------------
    # BACKWARD PASS
    # --------------------------------------------------------

    optimizer.zero_grad()

    loss.backward()


    # --------------------------------------------------------
    # GRADIENT CLIPPING
    # --------------------------------------------------------

    torch.nn.utils.clip_grad_norm_(

        model.parameters(),

        max_norm=1.0
    )


    # --------------------------------------------------------
    # OPTIMIZER UPDATE
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # RECORD LOSS
    # --------------------------------------------------------

    loss_value = loss.item()

    losses.append(loss_value)


    print(
        "Loss:",
        loss_value
    )

    print(
        "LoRA weights updated."
    )


    # --------------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------------

    print_gpu_memory(
        f"After step {step + 1}"
    )


    # --------------------------------------------------------
    # CLEAN GPU CACHE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# TRAINING FINISHED
# ============================================================

training_time = time.time() - start_time


print("\n========== TRAINING COMPLETE ==========")

print(
    "Samples trained:",
    NUM_SAMPLES
)

print(
    "Initial loss:",
    losses[0]
)

print(
    "Final loss:",
    losses[-1]
)

print(
    "Average loss:",
    sum(losses) / len(losses)
)

print(
    "Training time:",
    round(training_time, 2),
    "seconds"
)


# ============================================================
# SAVE LoRA ADAPTER
# ============================================================

print("\nSaving LoRA adapter...")

model.save_pretrained(
    OUTPUT_DIR
)

processor.save_pretrained(
    OUTPUT_DIR
)


print(
    "Adapter saved to:",
    OUTPUT_DIR
)


# ============================================================
# FINAL GPU MEMORY
# ============================================================

print_gpu_memory(
    "Final"
)


if torch.cuda.is_available():

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print(
        "\nFinal GPU peak memory:",
        round(peak_memory, 2),
        "GB"
    )


# ============================================================
# SUCCESS
# ============================================================

print("\n========== MINI TRAINING SUCCESS ==========")

print(
    "QLoRA training completed."
)

print(
    "LoRA adapter saved successfully."
)

print(
    "===========================================\n"
)