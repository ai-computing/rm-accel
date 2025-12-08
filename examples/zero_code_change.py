"""
Zero Code Change Example

This file demonstrates a script that can run on different backends
WITHOUT ANY MODIFICATION to the script itself.

This script is written as if it only uses vLLM. But by running it
through rm_abstract, it will automatically work on:
- NVIDIA GPU (vLLM)
- Rebellions NPU (vLLM-RBLN)
- FuriosaAI NPU (FuriosaAI SDK)
- CPU (Transformers fallback)

Run with different devices:
    # Auto-select best available device
    python -m rm_abstract.run examples/zero_code_change.py

    # Force specific device
    python -m rm_abstract.run examples/zero_code_change.py --device=cpu
    python -m rm_abstract.run examples/zero_code_change.py --device=nvidia
    python -m rm_abstract.run examples/zero_code_change.py --device=rbln

    # With verbose output
    python -m rm_abstract.run examples/zero_code_change.py --device=auto -v
"""

# =============================================================================
# ORIGINAL CODE - NO MODIFICATIONS NEEDED
# This is exactly how you would write vLLM code normally
# =============================================================================

from vllm import LLM, SamplingParams

def main():
    # Standard vLLM initialization
    llm = LLM(
        model="gpt2",
        trust_remote_code=True,
    )

    # Standard sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )

    # Standard prompts
    prompts = [
        "The future of AI is",
        "Once upon a time",
        "Python is a programming language that",
    ]

    # Standard generation
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # Standard output handling
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"\n[Prompt {i+1}] {prompt}")
        print(f"[Generated] {generated}")


if __name__ == "__main__":
    main()
