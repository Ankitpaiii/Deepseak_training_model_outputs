import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
import os

MODEL_DIR = r"D:\training_deepseak\model\deepseek-llm-7b-chat"
OUT_DIR = r"D:\training_deepseak\model\deepseek-llm-7b-chat-safetensors"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="cpu",   # IMPORTANT: CPU only for conversion
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

print("Saving safetensors...")
save_file(model.state_dict(), f"{OUT_DIR}\\model.safetensors")

tokenizer.save_pretrained(OUT_DIR)

print("✅ Conversion to safetensors completed")
