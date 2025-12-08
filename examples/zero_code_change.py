"""
Minimal Code Change Example

This example demonstrates how to migrate from vLLM to rm_abstract
with just ONE line of code change.

Before (vLLM only):
    from vllm import LLM, SamplingParams

After (rm_abstract - works on GPU, NPU, CPU):
    from rm_abstract import LLM, SamplingParams

That's it! The rest of the code remains exactly the same.

Run with different devices:
    # Auto-select best available device (NPU > GPU > CPU)
    python examples/zero_code_change.py

    # Force specific device via environment variable
    RM_DEVICE=cpu python examples/zero_code_change.py
    RM_DEVICE=nvidia python examples/zero_code_change.py
    RM_DEVICE=rbln python examples/zero_code_change.py
"""

# =============================================================================
# ONE LINE CHANGE - From vLLM to rm_abstract
# =============================================================================
# Before: from vllm import LLM, SamplingParams
from rm_abstract import LLM, SamplingParams  # <-- Only this line changes!


def main():
    # Standard LLM initialization - NO CHANGES NEEDED
    llm = LLM(
        model="gpt2",
        trust_remote_code=True,
    )

    # Show which backend is being used
    print(f"Backend: {llm.backend}")
    print(f"Device: {llm.device}")
    print(f"Engine: {llm.engine_type}")
    print()

    # Standard sampling parameters - NO CHANGES NEEDED
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )

    # Standard prompts - NO CHANGES NEEDED
    prompts = [
        "The future of AI is",
        "Once upon a time",
        "Python is a programming language that",
    ]

    # Standard generation - NO CHANGES NEEDED
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # Standard output handling - NO CHANGES NEEDED
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"\n[Prompt {i+1}] {prompt}")
        print(f"[Generated] {generated}")


if __name__ == "__main__":
    main()
