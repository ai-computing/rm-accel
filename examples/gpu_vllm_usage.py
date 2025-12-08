"""
GPU (vLLM) Usage Example with Device Switching

Demonstrates how to use RM Abstract Layer with GPU/vLLM backend
and switch to CPU backend at runtime.

Requirements:
    pip install torch vllm transformers

Usage:
    python examples/gpu_vllm_usage.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_backends():
    """Check available backends"""
import rm_abstract

print("=" * 60)
print("Checking available backends...")
print("=" * 60)

backends = rm_abstract.get_available_backends()
for name, available in backends.items():
    status = "✓ Available" if available else "✗ Not available"
    print(f"  {name}: {status}")
print()

    return backends


def run_inference(model, tokenizer, prompt, device_name):
    """Run inference and display results"""
    print(f"\n  [Input]  \"{prompt}\"")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  [Output] \"{generated_text}\"")
    
    return generated_text


def example1_gpu_inference():
    """Example 1: GPU inference with vLLM"""
    import rm_abstract
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 60)
    print("Example 1: GPU Inference (vLLM Backend)")
    print("=" * 60)

    # Initialize with GPU
    rm_abstract.init(device="auto", verbose=True)
    
    # Check current device
    info = rm_abstract.get_device_info()
    print(f"\n  Current Device: {info.get('device_type')}:{info.get('device_id')}")
    print(f"  GPU Name: {info.get('name', 'N/A')}")

    # Load model
    model_name = "gpt2"
    print(f"\n  Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"  Model type: {type(model).__name__}")

    # Run inference on GPU
    run_inference(model, tokenizer, "The future of AI is", "GPU")
    
    print()
    return model_name, tokenizer


def example2_switch_to_cpu(model_name, tokenizer):
    """Example 2: Switch to CPU and run inference"""
    import rm_abstract
    from transformers import AutoModelForCausalLM
    
    print("=" * 60)
    print("Example 2: Switch to CPU Backend")
    print("=" * 60)
    
    # Get current device before switching
    info_before = rm_abstract.get_device_info()
    print(f"\n  Before switch: {info_before.get('device_type')}:{info_before.get('device_id')}")
    
    # Switch to CPU
    print("\n  Switching device to CPU...")
    rm_abstract.switch_device("cpu")
    
    # Check device after switching
    info_after = rm_abstract.get_device_info()
    print(f"  After switch: {info_after.get('device_type')}:{info_after.get('device_id')}")
    
    # Load model on CPU
    print(f"\n  Loading model on CPU: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"  Model type: {type(model).__name__}")
    
    # Run inference on CPU
    run_inference(model, tokenizer, "The future of AI is", "CPU")
    
    print()


def example3_compare_outputs():
    """Example 3: Compare GPU vs CPU outputs side by side"""
    import rm_abstract
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 60)
    print("Example 3: GPU vs CPU Output Comparison")
    print("=" * 60)
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Machine learning enables computers to"
    
    # GPU inference
    print("\n  --- GPU (vLLM) ---")
    rm_abstract.init(device="gpu:0", verbose=False)
    model_gpu = AutoModelForCausalLM.from_pretrained(model_name)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs_gpu = model_gpu.generate(**inputs, max_new_tokens=25, do_sample=False)
    text_gpu = tokenizer.decode(outputs_gpu[0], skip_special_tokens=True)
    
    print(f"  Input:  \"{prompt}\"")
    print(f"  Output: \"{text_gpu}\"")
    
    # Switch to CPU
    print("\n  --- CPU (PyTorch) ---")
    rm_abstract.switch_device("cpu")
    model_cpu = AutoModelForCausalLM.from_pretrained(model_name)
    
    outputs_cpu = model_cpu.generate(**inputs, max_new_tokens=25, do_sample=False)
    text_cpu = tokenizer.decode(outputs_cpu[0], skip_special_tokens=True)
    
    print(f"  Input:  \"{prompt}\"")
    print(f"  Output: \"{text_cpu}\"")
    
    # Compare
    print("\n  --- Comparison ---")
    if text_gpu == text_cpu:
        print("  ✓ Outputs are identical!")
    else:
        print("  ⚠ Outputs differ (expected due to different backends)")

print()


def example4_device_info():
    """Example 4: Display device information"""
    import rm_abstract
    
print("=" * 60)
    print("Example 4: Final Device Information")
print("=" * 60)

    info = rm_abstract.get_device_info()
    print(f"  Device Type: {info.get('device_type', 'N/A')}")
    print(f"  Device ID: {info.get('device_id', 'N/A')}")
    print(f"  Name: {info.get('name', 'N/A')}")
    print(f"  Vendor: {info.get('vendor', 'N/A')}")
    if info.get("memory_total"):
        print(f"  Memory: {info['memory_total'] / 1e9:.1f} GB")

print()


def main():
    """Main entry point"""
    backends = check_backends()
    
    # Check if GPU is available
    if not backends.get("gpu", False):
        print("!" * 60)
        print("GPU/vLLM is NOT available. Running CPU-only test.")
        print("!" * 60)
        
        import rm_abstract
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        rm_abstract.init(device="cpu", verbose=True)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        run_inference(model, tokenizer, "The future of AI is", "CPU")
        return
    
    # Run examples
    try:
        model_name, tokenizer = example1_gpu_inference()
        example2_switch_to_cpu(model_name, tokenizer)
        example3_compare_outputs()
        example4_device_info()
except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("=" * 60)
    print("All Examples Completed!")
print("=" * 60)


# IMPORTANT: This guard is required for vLLM's multiprocessing (spawn method)
if __name__ == "__main__":
    main()
