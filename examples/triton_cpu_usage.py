"""
Triton CPU Usage Example

Same functionality as gpu_vllm_usage.py but using Triton on CPU.
Demonstrates how easy it is to switch serving engines.

Requirements:
    - Docker installed and running
    - Custom Triton image built: docker build -t rm-triton-server:latest -f docker/Dockerfile.triton docker/

Usage:
    python examples/triton_cpu_usage.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_requirements():
    """Check if Triton requirements are met"""
    from rm_abstract.serving.utils import is_docker_available
    
    print("=" * 60)
    print("Checking Triton Requirements")
    print("=" * 60)
    
    docker_ok = is_docker_available()
    print(f"  Docker: {'✓ Available' if docker_ok else '✗ Not available'}")
    
    if not docker_ok:
        print("\n  ⚠️ Docker is required for Triton.")
        print("  Run: sudo usermod -aG docker $USER && newgrp docker")
        return False
    
    return True


def run_triton_cpu():
    """Run inference with Triton on CPU"""
    from rm_abstract.serving import (
        create_serving_engine,
        ServingConfig,
        ServingEngineType,
        DeviceTarget,
    )
    
    print("\n" + "=" * 60)
    print("Triton CPU Inference")
    print("=" * 60)
    
    # Configure Triton with CPU
    config = ServingConfig(
        engine=ServingEngineType.TRITON,  # vLLM → Triton
        device=DeviceTarget.CPU,           # GPU → CPU
        model_name="gpt2",
        port=8000,
    )
    
    print(f"\n  Engine: {config.engine.value}")
    print(f"  Device: {config.device.value}")
    print(f"  Model:  {config.model_name}")
    
    # Use context manager for automatic server management
    try:
        with create_serving_engine(config) as engine:
            print("\n  ✓ Triton server started")
            
            # Setup model repository
            engine.setup_model_repository()
            print("  ✓ Model repository configured")
            
            # Load model
            engine.load_model("gpt2")
            print("  ✓ Model loaded")
            
            # Run inference
            prompts = [
                "Hello, I am",
                "The future of AI is",
            ]
            
            print("\n  Running inference...")
            for prompt in prompts:
                print(f"\n  [Input]  \"{prompt}\"")
                result = engine.infer(prompt, max_tokens=30)
                print(f"  [Output] \"{result}\"")
            
            print("\n  ✓ Inference complete")
            
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        print("\n  Troubleshooting:")
        print("  1. Ensure Docker is running: docker ps")
        print("  2. Build Triton image: docker build -t rm-triton-server:latest -f docker/Dockerfile.triton docker/")
        return False
    
    return True


def compare_with_vllm():
    """Show the difference between vLLM GPU and Triton CPU code"""
    print("\n" + "=" * 60)
    print("Code Comparison: vLLM GPU vs Triton CPU")
    print("=" * 60)
    
    print("""
  vLLM GPU (gpu_vllm_usage.py):
  ────────────────────────────
    import rm_abstract
    rm_abstract.init(device="gpu:0")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    outputs = model.generate(inputs)

  Triton CPU (this file):
  ────────────────────────
    config = ServingConfig(
        engine_type=ServingEngineType.TRITON,
        device=DeviceTarget.CPU,
    )
    with create_serving_engine(config) as engine:
        result = engine.infer(prompt)

  Key Differences:
  ────────────────
    ✓ Engine: vLLM → Triton (1 line)
    ✓ Device: GPU → CPU (1 line)
    ✓ API: HuggingFace → ServingEngine (different interface)
""")


def main():
    print("\n" + "=" * 60)
    print("  RM Abstract Layer - Triton CPU Example")
    print("=" * 60 + "\n")
    
    # Check requirements
    if not check_requirements():
        print("\n  Please install Docker and try again.")
        return
    
    # Show comparison
    compare_with_vllm()
    
    # Ask user
    print("\n  Run Triton CPU inference? (requires Docker)")
    print("  This will start a Triton server container.")
    response = input("  Continue? [y/N]: ").strip().lower()
    
    if response == 'y':
        run_triton_cpu()
    else:
        print("\n  Skipped. You can run manually with:")
        print("    python examples/triton_cpu_usage.py")


if __name__ == "__main__":
    main()

