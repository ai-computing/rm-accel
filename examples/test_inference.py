"""
Basic Inference Test

Test rm_abstract with different device/engine combinations.

Usage:
    python examples/test_inference.py
    python examples/test_inference.py --device cpu
    python examples/test_inference.py --device nvidia
    python examples/test_inference.py --device rbln
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Test inference with rm_abstract")
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
    args = parser.parse_args()

    # Configure rm_abstract BEFORE importing vllm
    from rm_abstract.auto import use, get_config

    use(device=args.device, engine=args.engine)

    print("=" * 60)
    print("RM Abstract Inference Test")
    print("=" * 60)
    print(f"Config: {get_config()}")
    print(f"Model: {args.model}")
    print()

    # Now import vllm - it will use the configured backend
    from vllm import LLM, SamplingParams

    print("Loading model...")
    llm = LLM(model=args.model)
    print(f"Backend: {llm.backend}")
    print()

    # Test generation
    prompts = [
        "The future of artificial intelligence is",
        "Python is a programming language that",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=50,
    )

    print("Generating...")
    print("-" * 60)

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"\n[Prompt {i+1}]")
        print(f"  Input: {output.prompt}")
        print(f"  Output: {output.outputs[0].text.strip()}")

    print()
    print("=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
