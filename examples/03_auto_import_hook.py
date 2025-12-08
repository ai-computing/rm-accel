#!/usr/bin/env python3
"""
Auto Import Hook Example - Transparent vLLM import hooking
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Configure device BEFORE importing vllm
from rm_abstract.auto import use

print("="*80)
print("CPU with vLLM Hook Example")
print("="*80)

# Configure to use CPU with vLLM hooking
use(device="cpu", engine="vllm", hook_vllm=True)

# Now import vllm - it will be automatically hooked to use transformers
from vllm import LLM, SamplingParams

print("Imported vLLM (actually using rm_abstract compat layer)")

# Use exactly the same code as you would with vLLM
llm = LLM(model="gpt2")

outputs = llm.generate(
    ["Hello, I am", "The future of AI"],
    SamplingParams(max_tokens=30, temperature=0.8)
)

for output in outputs:
    print(f"\nPrompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")

print("\n" + "="*80)
print("GPU with NO Hook Example (use original vLLM)")
print("="*80)

# For GPU, hooks are auto-disabled
use(device="nvidia", engine="vllm")

# Check hook status
from rm_abstract.auto import get_config, is_hooked
config = get_config()
print(f"Config: {config}")
print(f"Hooks enabled: {is_hooked()}")

print("""
When device is GPU/NPU:
- hook_vllm is automatically set to False
- Original vLLM library is used directly
- No compatibility layer overhead
""")
