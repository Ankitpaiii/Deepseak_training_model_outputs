import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from pathlib import Path

# =========================================================
# PATHS  (use LOCAL model — no internet download needed)
# =========================================================
BASE_DIR = Path(r"D:\training_deepseak")

MODEL_ID   = str(BASE_DIR / "model" / "deepseek-llm-7b-chat")
DATA_PATH  = BASE_DIR / "dataset" / "data.jsonl"
OUTPUT_DIR = BASE_DIR / "output" / "deepseek-lora"

# =========================================================
# TOKENIZER
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# MODEL (QLoRA — 4-bit via BitsAndBytesConfig)
# =========================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True
)

model.config.use_cache = False  # Required for gradient checkpointing

# =========================================================
# LoRA
# =========================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================================================
# DATASET
# =========================================================
dataset = load_dataset(
    "json",
    data_files=str(DATA_PATH),
    split="train"
)

def format_chat(example):
    """Format using the DeepSeek chat template from tokenizer_config.json."""
    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""
    text = ""
    for msg in example["messages"]:
        if msg["role"] == "system":
            text += f"{msg['content']}\n\n"
        elif msg["role"] == "user":
            text += f"User: {msg['content']}\n\n"
        elif msg["role"] == "assistant":
            text += f"Assistant: {msg['content']}{eos}"
    return f"{bos}{text}"

# =========================================================
# TRAINING ARGS  (SFTConfig for TRL 0.27+, WINDOWS SAFE)
# =========================================================
training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,                      # Disabled: BFloat16 model + fp16 AMP conflict
    bf16=False,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    max_grad_norm=0.3,               # Fixed: was 0.0 (disabled)
    max_length=512,              # Prevent OOM on 8 GB VRAM
    report_to="none",
    gradient_checkpointing=True,     # Save VRAM
    dataset_text_field=None,         # We use formatting_func
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=format_chat,
    peft_config=None,                # Already applied above
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ DeepSeek LoRA training completed successfully")
