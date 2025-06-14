import os
import sys
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/mnt/shareddata/models/deepSeek-R1-Distill-Qwen-32B/"
DEVICE = "cuda"  # ROCm-compatible PyTorch device

def is_rocm_available():
    return torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip is not None

def validate_environment():
    if not is_rocm_available():
        print("ERROR: ROCm environment not detected.")
        print("This script requires AMD ROCm-compatible GPUs and ROCm-enabled PyTorch.")
        sys.exit(1)

def print_device_info():
    visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", "all available")
    print(f"Using AMD ROCm device(s): {visible_devices}")

def load_model_and_tokenizer(model_dir):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        sys.exit(1)

def get_rocm_smi_stats():
    try:
        cmd = "rocm-smi --showuse --showmeminfo vram --showtemp"
        result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout if result.returncode == 0 else f"Could not retrieve ROCm-SMI stats.\n{result.stderr}"
    except Exception as e:
        return f"Error running rocm-smi: {e}"

def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")

def generate_response(prompt, tokenizer, model):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        return full_response[len(prompt):].strip().split("You:")[0].strip()
    except Exception as e:
        return f"[Error generating response: {e}]"

def interactive_chat_loop(tokenizer, model):
    chat_history = []
    while True:
        clear_screen()
        print("=" * 30 + " GPU Monitor " + "=" * 30)
        print(get_rocm_smi_stats())
        print("=" * 72)
        print("DeepSeek-Qwen Chat (type 'exit' to quit)\n")
        for line in chat_history:
            print(line)

        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break
        if not user_input:
            continue

        chat_history.append(f"You: {user_input}")
        prompt = "\n".join(chat_history) + "\nAssistant:"
        response = generate_response(prompt, tokenizer, model)
        chat_history.append(f"Assistant: {response}")

def batch_prompt_mode(tokenizer, model):
    print("Enter your prompts (empty line to finish):")
    prompts = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        prompts.append(line)

    if not prompts:
        print("No prompts entered.")
        return

    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    except Exception as e:
        print(f"Error moving inputs to device: {e}")
        return

    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id
            )
            responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
                print(f"\nPrompt {i}: {prompt}\nResponse: {response}\n")
        except Exception as e:
            print(f"Error during inference: {e}")

def main():
    validate_environment()
    print_device_info()
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)

    mode = input("Choose mode: (1) Batch prompts (2) Chat [1/2]: ").strip()
    if mode == "1":
        batch_prompt_mode(tokenizer, model)
    else:
        interactive_chat_loop(tokenizer, model)

if __name__ == "__main__":
    main()

cat: cat: No such file or directory
import os
import sys
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/mnt/shareddata/models/deepSeek-R1-Distill-Qwen-32B/"
DEVICE = "cuda"  # ROCm-compatible PyTorch device

def is_rocm_available():
    return torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip is not None

def validate_environment():
    if not is_rocm_available():
        print("ERROR: ROCm environment not detected.")
        print("This script requires AMD ROCm-compatible GPUs and ROCm-enabled PyTorch.")
        sys.exit(1)

def print_device_info():
    visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", "all available")
    print(f"Using AMD ROCm device(s): {visible_devices}")

def load_model_and_tokenizer(model_dir):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        sys.exit(1)

def get_rocm_smi_stats():
    try:
        cmd = "rocm-smi --showuse --showmeminfo vram --showtemp"
        result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout if result.returncode == 0 else f"Could not retrieve ROCm-SMI stats.\n{result.stderr}"
    except Exception as e:
        return f"Error running rocm-smi: {e}"

def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")

def generate_response(prompt, tokenizer, model):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        return full_response[len(prompt):].strip().split("You:")[0].strip()
    except Exception as e:
        return f"[Error generating response: {e}]"

def interactive_chat_loop(tokenizer, model):
    chat_history = []
    while True:
        clear_screen()
        print("=" * 30 + " GPU Monitor " + "=" * 30)
        print(get_rocm_smi_stats())
        print("=" * 72)
        print("DeepSeek-Qwen Chat (type 'exit' to quit)\n")
        for line in chat_history:
            print(line)

        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break
        if not user_input:
            continue

        chat_history.append(f"You: {user_input}")
        prompt = "\n".join(chat_history) + "\nAssistant:"
        response = generate_response(prompt, tokenizer, model)
        chat_history.append(f"Assistant: {response}")

def batch_prompt_mode(tokenizer, model):
    print("Enter your prompts (empty line to finish):")
    prompts = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        prompts.append(line)

    if not prompts:
        print("No prompts entered.")
        return

    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    except Exception as e:
        print(f"Error moving inputs to device: {e}")
        return

    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id
            )
            responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
                print(f"\nPrompt {i}: {prompt}\nResponse: {response}\n")
        except Exception as e:
            print(f"Error during inference: {e}")

def main():
    validate_environment()
    print_device_info()
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)

    mode = input("Choose mode: (1) Batch prompts (2) Chat [1/2]: ").strip()
    if mode == "1":
        batch_prompt_mode(tokenizer, model)
    else:
        interactive_chat_loop(tokenizer, model)

if __name__ == "__main__":
    main()
