import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
from peft import PeftModel
import sys

# =========================================================
# CONFIGURATION
# =========================================================
BASE_MODEL_ID = r"D:\training_deepseak\model\deepseek-llm-7b-chat"
ADAPTER_PATH  = r"D:\training_deepseak\output\deepseek-lora"

# =========================================================
# LOAD MODEL & TOKENIZER
# =========================================================
print(f"Loading base model from: {BASE_MODEL_ID}")

# Using 4-bit quantization to match training environment and save memory
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
# INFERENCE LOOP
# =========================================================
print("\n" + "="*50)
print("🤖 Model Loaded Successfully!")
print("Type 'exit' or 'quit' to stop.")
print("="*50 + "\n")

def generate_response(user_input):
    # Format matches the training data format:
    # User: {content}\n\nAssistant: {content}
    prompt = f"User: {user_input}\n\nAssistant:"
    
    if tokenizer.bos_token:
        prompt = tokenizer.bos_token + prompt

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Streamer allows seeing the output token by token
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("Assistant: ", end="", flush=True)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    print("\n")

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        if not user_input.strip():
            continue
            
        generate_response(user_input)
        print("-" * 30)
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"\nAn error occurred during generation: {e}")
