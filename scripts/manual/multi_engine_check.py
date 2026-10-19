#!/usr/bin/env python3
"""Test multi-engine functionality with GPU and CPU."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from rm_abstract import LLM, SamplingParams
from rm_abstract.auto import use


def test_cpu_vllm():
    """Test CPU with vLLM engine (should fallback to transformers via hook)."""
    print("\n" + "="*80)
    print("TEST 1: CPU with vLLM engine (hook enabled)")
    print("="*80)

    try:
        # Configure for CPU with vLLM hook
        use(device="cpu", engine="vllm", hook_vllm=True)

        llm = LLM(model="gpt2", device="cpu", engine="vllm")
        outputs = llm.generate(
            ["Hello, I am"],
            SamplingParams(max_tokens=20, temperature=0.0)
        )

        print(f"✓ Success!")
        print(f"  Input: 'Hello, I am'")
        print(f"  Output: {outputs[0].outputs[0].text}")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cpu_transformers():
    """Test CPU with transformers directly (no vLLM)."""
    print("\n" + "="*80)
    print("TEST 2: CPU with transformers directly")
    print("="*80)

    try:
        # Disable hooks to use transformers directly
        use(device="cpu", engine="vllm", hook_vllm=False)

        llm = LLM(model="gpt2", device="cpu")
        outputs = llm.generate(
            ["Hello, I am"],
            SamplingParams(max_tokens=20, temperature=0.0)
        )

        print(f"✓ Success!")
        print(f"  Input: 'Hello, I am'")
        print(f"  Output: {outputs[0].outputs[0].text}")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_vllm():
    """Test GPU with vLLM engine (original vLLM)."""
    print("\n" + "="*80)
    print("TEST 3: GPU with vLLM engine (hook disabled)")
    print("="*80)

    try:
        # Configure for GPU (hooks auto-disabled)
        # Use GPU 1 to avoid GPU 0 which is occupied by TorchServe
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        use(device="nvidia", engine="vllm")

        llm = LLM(
            model="gpt2",
            device="nvidia",
            engine="vllm",
            enforce_eager=True,
            disable_log_stats=True
        )
        outputs = llm.generate(
            ["Hello, I am"],
            SamplingParams(max_tokens=20, temperature=0.0)
        )

        print(f"✓ Success!")
        print(f"  Input: 'Hello, I am'")
        print(f"  Output: {outputs[0].outputs[0].text}")
        return True

    except Exception as e:
        print(f"✗ Failed (GPU may not be available): {e}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_import_cpu():
    """Test auto-import with CPU (import vllm after use())."""
    print("\n" + "="*80)
    print("TEST 4: Auto-import with CPU (hook vllm)")
    print("="*80)

    try:
        # Reset modules
        if 'vllm' in sys.modules:
            del sys.modules['vllm']

        # Configure BEFORE importing vllm
        use(device="cpu", engine="vllm", hook_vllm=True)

        # Now import vllm - should be hooked
        from vllm import LLM as VllmLLM, SamplingParams as VllmSamplingParams

        llm = VllmLLM(model="gpt2")
        outputs = llm.generate(
            ["Hello, I am"],
            VllmSamplingParams(max_tokens=20, temperature=0.0)
        )

        print(f"✓ Success!")
        print(f"  Input: 'Hello, I am'")
        print(f"  Output: {outputs[0].outputs[0].text}")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_import_gpu():
    """Test auto-import with GPU (import vllm after use())."""
    print("\n" + "="*80)
    print("TEST 5: Auto-import with GPU (no hook)")
    print("="*80)

    try:
        # Reset modules
        if 'vllm' in sys.modules:
            del sys.modules['vllm']

        # Use GPU 2 to avoid conflicts
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"

        # Configure BEFORE importing vllm
        use(device="nvidia", engine="vllm")

        # Now import vllm - should NOT be hooked (use original)
        from vllm import LLM as VllmLLM, SamplingParams as VllmSamplingParams

        llm = VllmLLM(
            model="gpt2",
            enforce_eager=True,
            disable_log_stats=True
        )
        outputs = llm.generate(
            ["Hello, I am"],
            VllmSamplingParams(max_tokens=20, temperature=0.0)
        )

        print(f"✓ Success!")
        print(f"  Input: 'Hello, I am'")
        print(f"  Output: {outputs[0].outputs[0].text}")
        return True

    except Exception as e:
        print(f"✗ Failed (GPU may not be available): {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MULTI-ENGINE GPU/CPU TEST SUITE")
    print("="*80)

    results = {}

    # Test CPU configurations
    results['cpu_vllm'] = test_cpu_vllm()
    results['cpu_transformers'] = test_cpu_transformers()
    results['auto_import_cpu'] = test_auto_import_cpu()

    # Test GPU configurations (may fail if no GPU)
    results['gpu_vllm'] = test_gpu_vllm()
    results['auto_import_gpu'] = test_auto_import_gpu()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
