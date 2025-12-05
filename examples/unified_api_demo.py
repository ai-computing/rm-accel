"""
Unified API Demo - Same Code, Different Engines

This example shows how the SAME code can run on:
- vLLM GPU
- Triton CPU
- TorchServe CPU
- HuggingFace CPU

Just change the engine and device parameters!

Usage:
    # Default (auto-select)
    python examples/unified_api_demo.py
    
    # Specific engine
    python examples/unified_api_demo.py --engine vllm --device gpu
    python examples/unified_api_demo.py --engine triton --device cpu
    python examples/unified_api_demo.py --engine huggingface --device cpu
"""

import sys
import argparse
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_inference(engine: str, device: str):
    """
    Run inference using UnifiedModel.
    
    THE SAME CODE works for ALL engines!
    """
    from rm_abstract import UnifiedModel
    
    print("=" * 60)
    print(f"  Engine: {engine}, Device: {device}")
    print("=" * 60)
    
    # Create model - same API for all engines!
    print(f"\n  Creating UnifiedModel...")
    model = UnifiedModel(
        "gpt2",
        engine=engine,
        device=device,
    )
    print(f"  ✓ Model created: {model}")
    print(f"  ✓ Actual engine: {model.engine_name}")
    
    # Generate - same API for all engines!
    prompts = [
        "Hello, I am",
        "The future of AI is",
    ]
    
    print("\n  Running inference...")
    for prompt in prompts:
        print(f"\n  [Input]  \"{prompt}\"")
        
        output = model.generate(
            prompt,
            max_tokens=30,
            temperature=0.7,
            do_sample=True,
        )
        
        print(f"  [Output] \"{output}\"")
    
    # Cleanup
    model.close()
    print("\n  ✓ Done!")


def compare_engines():
    """Show how the same code works across engines"""
    print("\n" + "=" * 60)
    print("  Code Comparison - SAME API for ALL engines!")
    print("=" * 60)
    
    print("""
  # vLLM GPU
  model = UnifiedModel("gpt2", engine="vllm", device="gpu")
  output = model.generate("Hello")
  
  # Triton CPU - SAME generate() call!
  model = UnifiedModel("gpt2", engine="triton", device="cpu")
  output = model.generate("Hello")
  
  # TorchServe CPU - SAME generate() call!
  model = UnifiedModel("gpt2", engine="torchserve", device="cpu")
  output = model.generate("Hello")
  
  # HuggingFace CPU - SAME generate() call!
  model = UnifiedModel("gpt2", engine="huggingface", device="cpu")
  output = model.generate("Hello")
  
  ✓ All use the SAME model.generate() API!
""")


def main():
    parser = argparse.ArgumentParser(description="Unified API Demo")
    parser.add_argument("--engine", default="huggingface",
                       choices=["auto", "vllm", "triton", "torchserve", "huggingface"],
                       help="Serving engine to use")
    parser.add_argument("--device", default="cpu",
                       choices=["auto", "gpu", "cpu"],
                       help="Device to run on")
    parser.add_argument("--compare", action="store_true",
                       help="Show code comparison")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  RM Abstract Layer - Unified API Demo")
    print("=" * 60)
    
    if args.compare:
        compare_engines()
        return
    
    try:
        run_inference(args.engine, args.device)
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        print("\n  Try with --engine huggingface --device cpu")
    
    print("\n" + "=" * 60)
    print("  Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

