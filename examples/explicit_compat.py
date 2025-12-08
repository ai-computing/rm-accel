"""
Explicit Device Selection Example

Demonstrates explicit control over which device and engine is used.
This is the recommended way to use rm_abstract.

Usage:
    python examples/explicit_compat.py
"""

from rm_abstract import LLM, SamplingParams

print("=" * 60)
print("Explicit Device Selection Example")
print("=" * 60)


def example_vllm_style():
    """Example using vLLM-style API."""
    print("\n--- vLLM-style API ---")

    # Explicitly specify the device
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


def example_with_engine():
    """Example with specific engine selection."""
    print("\n--- Engine Selection ---")

    # Specify both device and engine
    llm = LLM(
        model="gpt2",
        device="cpu",
        engine="vllm",  # or "triton", "torchserve"
    )

    print(f"Backend: {llm.backend}")
    print(f"Engine: {llm.engine_type}")

    try:
        outputs = llm.generate(
            ["The capital of France is"],
            SamplingParams(max_tokens=20)
        )
        print(f"Generated: {outputs[0].outputs[0].text}")
    except Exception as e:
        print(f"Generation skipped: {e}")


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
    elif "RM_DEVICE" in os.environ:
        del os.environ["RM_DEVICE"]


if __name__ == "__main__":
    example_vllm_style()
    example_with_engine()
    example_with_env()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
