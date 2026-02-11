import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# =========================================================
# CONFIGURATION
# =========================================================
BASE_MODEL_ID = r"D:\training_deepseak\model\deepseek-llm-7b-chat"
ADAPTER_PATH  = r"D:\training_deepseak\output\deepseek-lora"

# Sample prompts for testing the refining capability
# TEST_PROMPTS = [
#     "Refine this prompt: write a story about a cat",
#     "Improve this prompt: I want a python script for web scraping",
#     "Optimize: tell me about black holes",
#     "Make this better: create a marketing email",
#     "Refine: explain quantum physics to a 5 year old"
# ]
TEST_PROMPTS = [
    "Improve this prompt: write a story about a cat",
    "Improve this prompt: I want a python script for web scraping",
    "Improve this prompt: tell me about black holes",
    "Improve this prompt: create a marketing email",
    "Improve this prompt: explain quantum physics to a 5 year old"
]


# =========================================================
# LOAD MODEL & TOKENIZER
# =========================================================
print(f"Loading base model from: {BASE_MODEL_ID}")

# Check if model paths exist
if not os.path.exists(BASE_MODEL_ID):
    print(f"Error: Base model path not found: {BASE_MODEL_ID}")
    sys.exit(1)
if not os.path.exists(ADAPTER_PATH):
    print(f"Error: Adapter path not found: {ADAPTER_PATH}")
    sys.exit(1)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
except Exception as e:
    print(f"Error loading base model: {e}")
    sys.exit(1)

print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
try:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
except Exception as e:
    print(f"Error loading adapter. Ensure the path is correct and training finished successfully.\nError: {e}")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# INFERENCE
# =========================================================
print("\n" + "="*50)
print("Model Loaded. Starting Inference on Test Prompts...")
print("="*50 + "\n")

results = []

for i, user_input in enumerate(TEST_PROMPTS):
    print(f"Processing Prompt {i+1}/{len(TEST_PROMPTS)}: {user_input}")
    
    # Format matches the training data format:
    # User: {content}\n\nAssistant: {content}
    prompt = f"User: {user_input}\n\nAssistant:"
    
    if tokenizer.bos_token:
        prompt = tokenizer.bos_token + prompt

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    generated_text = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    result_entry = f"Original Prompt: {user_input}\nRefined Output: {generated_text}\n" + "-"*50
    results.append(result_entry)
    print(f"Result generated.\n")

# Save results
output_file = r"D:\training_deepseak\inference_results.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for res in results:
        f.write(res + "\n")

print(f"\nAll results saved to {output_file}")
