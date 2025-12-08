"""
Basic Usage Example

Demonstrates basic usage of RM Abstract Layer with the unified LLM interface.
"""

from rm_abstract import LLM, SamplingParams, get_available_resources


def main():
    # 1. Check available resources
    print("Available resources:")
    for name, available in get_available_resources().items():
        status = "OK" if available else "Not available"
        print(f"  {name}: {status}")

    # 2. Create LLM with auto-selection (NPU > GPU > CPU)
    print("\nInitializing LLM with auto device selection...")
    llm = LLM(model="gpt2", device="auto")
    print(f"  Model: {llm.model_name}")
    print(f"  Device: {llm.device}")
    print(f"  Backend: {llm.backend}")
    print(f"  Engine: {llm.engine_type}")

    # 3. Generate text
    print("\nGenerating text...")
    prompts = ["Hello, I am"]
    params = SamplingParams(max_tokens=30, temperature=0.8)

    outputs = llm.generate(prompts, params)

    for output in outputs:
        print(f"  Prompt: {output.prompt}")
        print(f"  Generated: {output.outputs[0].text}")


if __name__ == "__main__":
    main()
