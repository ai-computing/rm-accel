#!/usr/bin/env python3
"""
Device Selection Example - Explicit device control
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rm_abstract import LLM, SamplingParams


def test_cpu():
    """Test CPU inference"""
    print("\n" + "="*80)
    print("CPU Inference (transformers)")
    print("="*80)

    llm = LLM(model="gpt2", device="cpu")
    outputs = llm.generate(
        ["Hello, I am"],
        SamplingParams(max_tokens=30, temperature=0.0)
    )

    print(f"Output: {outputs[0].outputs[0].text}")


def test_gpu():
    """Test GPU inference"""
    print("\n" + "="*80)
    print("GPU Inference (vLLM)")
    print("="*80)

    try:
        # Use specific GPU to avoid memory conflicts
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        llm = LLM(
            model="gpt2",
            device="nvidia",
            enforce_eager=True,  # Faster initialization
            gpu_memory_utilization=0.15,  # Low memory for shared environments
        )
        outputs = llm.generate(
            ["Hello, I am"],
            SamplingParams(max_tokens=30, temperature=0.0)
        )

        print(f"Output: {outputs[0].outputs[0].text}")

    except Exception as e:
        print(f"GPU test failed: {e}")
        print("This is normal if GPU is not available")


def test_auto():
    """Test automatic device selection"""
    print("\n" + "="*80)
    print("Auto Device Selection (NPU > GPU > CPU)")
    print("="*80)

    from rm_abstract import get_available_resources

    # Check available resources
    resources = get_available_resources()
    print(f"Available resources: {resources}")

    # Create LLM with auto device selection
    llm = LLM(model="gpt2", device="auto")

    # Note: Device info is available after LLM initialization
    print(f"Using device: auto (NPU > GPU > CPU priority)")

    # Generate
    outputs = llm.generate(
        ["Hello, I am"],
        SamplingParams(max_tokens=30, temperature=0.0)
    )

    print(f"Output: {outputs[0].outputs[0].text}")


if __name__ == "__main__":
    test_cpu()
    test_auto()
    # test_gpu()  # Uncomment if GPU is available
