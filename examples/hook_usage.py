"""
Hook Usage Example

Demonstrates how to use rm_abstract hooks with explicit device/engine selection.

Usage Patterns:

1. Explicit selection (recommended):
    from rm_abstract.auto import use
    use(device="rbln", engine="vllm")

2. Environment variable:
    RM_DEVICE=rbln python examples/hook_usage.py

3. Launcher:
    python -m rm_abstract.run examples/hook_usage.py --device=rbln
"""

# =============================================================================
# Method 1: Explicit device and engine selection
# =============================================================================
from rm_abstract.auto import use

# Choose your device and engine before importing ML libraries
use(device="cpu", engine="vllm")  # CPU fallback
# use(device="nvidia", engine="vllm")  # NVIDIA GPU
# use(device="rbln", engine="vllm")    # Rebellions NPU
# use(device="furiosa", engine="vllm") # FuriosaAI NPU

# Now import vLLM as normal - it will use the selected backend
from vllm import LLM, SamplingParams

print("=" * 60)
print("RM Abstract Hook Usage Example")
print("=" * 60)

# Check configuration
from rm_abstract.auto import get_config
config = get_config()
print(f"\nConfiguration:")
print(f"  Device: {config['device']}")
print(f"  Engine: {config['engine']}")

# Create LLM instance
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

print("\n" + "=" * 60)
