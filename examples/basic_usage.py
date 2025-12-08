"""
Basic Usage Example

Demonstrates basic usage of RM Abstract Layer.

IMPORTANT: When using vLLM backend, code must be inside
if __name__ == "__main__": block due to multiprocessing requirements.
"""

import rm_abstract
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # 1. Initialize RM Abstract Layer
# Device settings:
# - "auto": Auto-select (NPU > GPU > CPU)
# - "gpu:0": First GPU
# - "rbln:0": Rebellions ATOM NPU
# - "cpu": CPU
rm_abstract.init(device="auto", verbose=True)

# 2. Use existing code as-is below
# Load model (automatically prepared for the backend)
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Execute inference
inputs = tokenizer("Hello, I am a language model,", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Check device info
print("\nDevice Info:")
print(rm_abstract.get_device_info())


# IMPORTANT: This guard is required for vLLM's multiprocessing (spawn method)
if __name__ == "__main__":
    main()
