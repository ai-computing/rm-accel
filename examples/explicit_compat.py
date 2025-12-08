"""
Explicit Compatibility Layer Usage

If you prefer explicit control over which backend is used,
you can import directly from rm_abstract.compat instead of using hooks.

This gives you more control and makes the backend switching explicit
in your code.
"""

# Import from rm_abstract.compat instead of vllm/transformers
from rm_abstract.compat import LLM, SamplingParams
from rm_abstract.compat import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("Explicit Compatibility Layer Example")
print("=" * 60)


def example_vllm_style():
    """Example using vLLM-style API."""
    print("\n--- vLLM-style API ---")

    # You can explicitly specify the device
    llm = LLM(
        model="gpt2",
        device="cpu",  # Explicit device selection
    )

    print(f"Backend: {llm.backend}")

    params = SamplingParams(
        temperature=0.8,
        max_tokens=30,
    )

    try:
        outputs = llm.generate(["Hello, world!"], params)
        print(f"Generated: {outputs[0].outputs[0].text}")
    except Exception as e:
        print(f"Generation skipped: {e}")


def example_transformers_style():
    """Example using Transformers-style API."""
    print("\n--- Transformers-style API ---")

    # Load tokenizer (device-agnostic)
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print(f"Tokenizer loaded: {type(tokenizer).__name__}")

        # Load model with explicit device
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            device="cpu",  # Explicit device selection
        )
        print(f"Model loaded: {type(model).__name__}")

        # Generate
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        outputs = model.generate(inputs.input_ids, max_new_tokens=20)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {text}")

    except Exception as e:
        print(f"Example skipped: {e}")


def example_with_env():
    """Example using environment variable for device selection."""
    print("\n--- Environment Variable Control ---")

    import os

    # Set device via environment variable
    original_device = os.environ.get("RM_DEVICE")
    os.environ["RM_DEVICE"] = "cpu"

    # LLM will use the environment variable
    llm = LLM(model="gpt2")
    print(f"Backend (from env): {llm.backend}")

    # Restore original
    if original_device:
        os.environ["RM_DEVICE"] = original_device
    else:
        del os.environ["RM_DEVICE"]


if __name__ == "__main__":
    example_vllm_style()
    example_transformers_style()
    example_with_env()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
