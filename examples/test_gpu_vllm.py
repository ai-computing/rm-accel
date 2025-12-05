"""
GPU/vLLM Backend Test Script

This script tests if the GPU/vLLM backend works correctly with the
"Zero Code Change" principle.

Requirements:
    - NVIDIA GPU with CUDA
    - pip install vllm transformers torch

Usage:
    python examples/test_gpu_vllm.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_backends_availability():
    """Test 1: Check available backends"""
    print("=" * 60)
    print("Test 1: Check available backends")
    print("=" * 60)

    import rm_abstract

    backends = rm_abstract.get_available_backends()
    print(f"Available backends: {backends}")

    gpu_available = backends.get("gpu", False)
    print(f"GPU/vLLM available: {gpu_available}")

    return gpu_available


def test_gpu_initialization():
    """Test 2: Initialize GPU backend"""
    print("\n" + "=" * 60)
    print("Test 2: Initialize GPU backend")
    print("=" * 60)

    import rm_abstract

    try:
        rm_abstract.init(device="gpu:0", verbose=True)
        info = rm_abstract.get_device_info()
        print(f"Device info: {info}")
        return True
    except Exception as e:
        print(f"Failed to initialize GPU: {e}")
        return False


def test_zero_code_change():
    """Test 3: Zero Code Change - HuggingFace style code"""
    print("\n" + "=" * 60)
    print("Test 3: Zero Code Change - HuggingFace style")
    print("=" * 60)

    import rm_abstract

    # Initialize (should already be done, but just in case)
    rm_abstract.init(device="gpu:0", verbose=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"Model type: {type(model).__name__}")

    # Test with tokenized input (HuggingFace style)
    print("\n--- Test with tokenized input ---")
    inputs = tokenizer("Hello, I am a language model", return_tensors="pt")
    print(f"Input tokens shape: {inputs['input_ids'].shape}")

    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"Output type: {type(outputs)}")
    print(f"Output shape: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")

    if hasattr(outputs, "shape"):
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {text}")

    return True


def test_string_prompt():
    """Test 4: Direct string prompt"""
    print("\n" + "=" * 60)
    print("Test 4: Direct string prompt (vLLM native style)")
    print("=" * 60)

    import rm_abstract

    rm_abstract.init(device="gpu:0", verbose=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Test with string prompt (vLLM native style)
    print("Testing with string prompt...")
    outputs = model.generate("What is artificial intelligence?", max_tokens=50)

    print(f"Output type: {type(outputs)}")
    if hasattr(outputs, "__iter__"):
        for i, output in enumerate(outputs):
            print(f"Output {i}: {output}")

    return True


def main():
    print("GPU/vLLM Backend Test Suite")
    print("=" * 60)

    # Test 1: Check backends
    gpu_available = test_backends_availability()

    if not gpu_available:
        print("\n" + "!" * 60)
        print("GPU/vLLM is NOT available.")
        print("This test requires:")
        print("  - NVIDIA GPU with CUDA")
        print("  - vllm package: pip install vllm")
        print("!" * 60)

        # Test with CPU fallback
        print("\nTesting with CPU backend instead...")
        import rm_abstract

        rm_abstract.init(device="cpu", verbose=True)


        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")

        inputs = tokenizer("Hello, I am", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        print(f"CPU Output: {tokenizer.decode(outputs[0])}")
        return

    # Test 2: Initialize GPU
    if not test_gpu_initialization():
        print("GPU initialization failed!")
        return

    # Test 3: Zero Code Change
    try:
        test_zero_code_change()
    except Exception as e:
        print(f"Test 3 failed: {e}")
        import traceback

        traceback.print_exc()

    # Test 4: String prompt
    try:
        test_string_prompt()
    except Exception as e:
        print(f"Test 4 failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
