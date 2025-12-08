"""
Engine Selection Example

Demonstrates how to use different serving engines with rm_abstract.

Supported engines:
- vllm: High-performance LLM serving (default)
- triton: NVIDIA Triton Inference Server
- torchserve: PyTorch TorchServe

Usage:
    python examples/engine_selection.py
    python examples/engine_selection.py --engine vllm --device cpu
    python examples/engine_selection.py --engine triton --device nvidia
"""

import argparse
from rm_abstract import LLM, SamplingParams


def test_engine(device: str, engine: str, model: str = "gpt2"):
    """Test inference with a specific device and engine."""
    print(f"\n{'='*60}")
    print(f"Testing: device={device}, engine={engine}")
    print(f"{'='*60}")

    try:
        print(f"\nLoading model: {model}")
        llm = LLM(model=model, device=device, engine=engine)
        print(f"Backend: {llm.backend}")
        print(f"Engine: {llm.engine_type}")

        # Generate
        params = SamplingParams(max_tokens=30, temperature=0.7)
        outputs = llm.generate(["Hello, I am"], params)

        text = outputs[0].outputs[0].text.strip().replace("\n", " ")[:60]
        print(f"Output: {text}...")

        return True, None

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Test different serving engines")
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        choices=["auto", "cpu", "nvidia", "rbln", "furiosa"],
        help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--engine", "-e",
        default="vllm",
        choices=["vllm", "triton", "torchserve"],
        help="Serving engine (default: vllm)"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt2",
        help="Model name (default: gpt2)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Test all available combinations"
    )
    args = parser.parse_args()

    if args.all:
        # Test all combinations
        print("="*60)
        print("Testing All Device/Engine Combinations")
        print("="*60)

        combinations = [
            ("cpu", "vllm"),
            ("cpu", "triton"),
            ("cpu", "torchserve"),
        ]

        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                combinations.extend([
                    ("nvidia", "vllm"),
                    ("nvidia", "triton"),
                    ("nvidia", "torchserve"),
                ])
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        except ImportError:
            pass

        results = {}
        for device, engine in combinations:
            success, error = test_engine(device, engine, args.model)
            results[(device, engine)] = (success, error)

        # Summary
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        for (device, engine), (success, error) in results.items():
            status = "PASS" if success else f"FAIL ({error[:30]}...)" if error else "FAIL"
            print(f"  {device:8} + {engine:12} : {status}")

    else:
        # Single test
        test_engine(args.device, args.engine, args.model)


if __name__ == "__main__":
    main()
