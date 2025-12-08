#!/usr/bin/env python3
"""
Serving Engines Example - Test different serving engines
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rm_abstract import LLM, SamplingParams


def test_vllm_engine():
    """Test vLLM serving engine"""
    print("\n" + "="*80)
    print("vLLM Engine (CPU with transformers fallback)")
    print("="*80)

    try:
        llm = LLM(
            model="gpt2",
            device="cpu",
            engine="vllm",
        )

        outputs = llm.generate(
            ["Hello, I am"],
            SamplingParams(max_tokens=20, temperature=0.0)
        )

        print(f"✓ Success!")
        print(f"  Output: {outputs[0].outputs[0].text}")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_torchserve_engine():
    """Test TorchServe serving engine"""
    print("\n" + "="*80)
    print("TorchServe Engine")
    print("="*80)

    try:
        llm = LLM(
            model="gpt2",
            device="cpu",
            engine="torchserve",
        )

        outputs = llm.generate(
            ["Hello, I am"],
            SamplingParams(max_tokens=20, temperature=0.0)
        )

        print(f"✓ Success!")
        print(f"  Output: {outputs[0].outputs[0].text}")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        print("Note: TorchServe requires 'torchserve' and 'torch-model-archiver' installed")
        print("  pip install torchserve torch-model-archiver")
        return False


def test_triton_engine():
    """Test Triton serving engine"""
    print("\n" + "="*80)
    print("Triton Engine")
    print("="*80)

    try:
        llm = LLM(
            model="gpt2",
            device="cpu",
            engine="triton",
        )

        outputs = llm.generate(
            ["Hello, I am"],
            SamplingParams(max_tokens=20, temperature=0.0)
        )

        print(f"✓ Success!")
        print(f"  Output: {outputs[0].outputs[0].text}")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        print("Note: Triton requires Docker and 'tritonclient' installed")
        print("  pip install tritonclient[http]")
        return False


def main():
    """Run all engine tests"""
    print("="*80)
    print("SERVING ENGINE TESTS")
    print("="*80)

    results = {}

    # Test each engine
    results['vllm'] = test_vllm_engine()
    results['torchserve'] = test_torchserve_engine()
    results['triton'] = test_triton_engine()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for engine, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {engine}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
