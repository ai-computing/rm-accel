#!/usr/bin/env python3
"""
Engine Comparison Example - Compare performance across engines
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rm_abstract import LLM, SamplingParams


def benchmark_engine(engine_name, device="cpu", model="gpt2"):
    """Benchmark a specific engine"""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {engine_name} on {device}")
    print(f"{'='*80}")

    try:
        # Initialize
        start_time = time.time()
        llm = LLM(
            model=model,
            device=device,
            engine=engine_name,
        )
        init_time = time.time() - start_time

        # Warmup
        llm.generate(["Warmup"], SamplingParams(max_tokens=5))

        # Benchmark
        prompts = ["Hello, I am"] * 5
        sampling_params = SamplingParams(
            max_tokens=20,
            temperature=0.0,
        )

        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        inference_time = time.time() - start_time

        # Results
        print(f"✓ Success!")
        print(f"  Initialization: {init_time:.2f}s")
        print(f"  Inference (5 prompts): {inference_time:.2f}s")
        print(f"  Per prompt: {inference_time/len(prompts):.2f}s")
        print(f"  Sample output: {outputs[0].outputs[0].text}")

        return {
            'success': True,
            'init_time': init_time,
            'inference_time': inference_time,
            'per_prompt': inference_time / len(prompts),
        }

    except Exception as e:
        print(f"✗ Failed: {e}")
        return {
            'success': False,
            'error': str(e),
        }


def main():
    """Compare engines"""
    print("="*80)
    print("ENGINE COMPARISON BENCHMARK")
    print("="*80)

    engines = ["vllm", "torchserve", "triton"]
    results = {}

    for engine in engines:
        results[engine] = benchmark_engine(engine, device="cpu")

    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    successful = {k: v for k, v in results.items() if v['success']}

    if successful:
        print(f"\n{'Engine':<15} {'Init (s)':<12} {'Inference (s)':<15} {'Per prompt (s)':<15}")
        print("-" * 60)

        for engine, data in successful.items():
            print(f"{engine:<15} {data['init_time']:<12.2f} {data['inference_time']:<15.2f} {data['per_prompt']:<15.2f}")

        # Find fastest
        fastest = min(successful.items(), key=lambda x: x[1]['inference_time'])
        print(f"\n🏆 Fastest inference: {fastest[0]} ({fastest[1]['inference_time']:.2f}s)")

    # Failed engines
    failed = {k: v for k, v in results.items() if not v['success']}
    if failed:
        print("\n❌ Failed engines:")
        for engine, data in failed.items():
            print(f"  - {engine}: {data['error']}")


if __name__ == "__main__":
    main()
