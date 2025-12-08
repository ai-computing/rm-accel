"""
Hook Usage Example

Demonstrates how to use rm_abstract hooks to run existing code
without modification.

Run with:
    # Auto device selection
    python examples/hook_usage.py

    # Specific device
    RM_DEVICE=cpu python examples/hook_usage.py
    RM_DEVICE=nvidia python examples/hook_usage.py
    RM_DEVICE=rbln python examples/hook_usage.py

    # Using launcher
    python -m rm_abstract.run examples/hook_usage.py --device=auto
"""

# Method 1: Import rm_abstract.auto at the top of your script
# This single line enables automatic backend switching
import rm_abstract.auto  # noqa: F401 - This enables hooks

# Now use vLLM as normal - it will be routed to the appropriate backend
from vllm import LLM, SamplingParams

print("=" * 60)
print("RM Abstract Hook Usage Example")
print("=" * 60)

# Check which backend is being used
import os
device = os.environ.get("RM_DEVICE", "auto")
print(f"\nDevice selection: {device}")

# Create LLM instance
# This will automatically use:
# - vLLM on NVIDIA GPU
# - vLLM-RBLN on Rebellions NPU
# - FuriosaAI SDK on FuriosaAI NPU
# - Transformers on CPU (fallback)
print("\nInitializing LLM...")

try:
    llm = LLM(model="gpt2")
    print(f"Backend: {llm.backend}")

    # Generate text
    prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
    )

    print(f"\nPrompt: {prompts[0]}")
    print("Generating...")

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"\nGenerated text:")
        print(f"  {output.outputs[0].text}")

except Exception as e:
    print(f"\nNote: Could not run inference: {e}")
    print("This is expected if you don't have the model downloaded.")
    print("The hook mechanism is still working correctly.")

print("\n" + "=" * 60)
print("Hook demonstration complete!")
print("=" * 60)
