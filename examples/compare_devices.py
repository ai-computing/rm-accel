"""
Compare Different Devices

Run the same inference on different devices and compare results.

Usage:
    python examples/compare_devices.py
"""

import sys
import time


def test_device(device: str, model: str = "gpt2"):
    """Test inference on a specific device."""
    # Clean module cache for fresh import
    for mod in list(sys.modules.keys()):
        if "rm_abstract" in mod or mod == "vllm":
            del sys.modules[mod]

    from rm_abstract.auto import use, get_config

    use(device=device, engine="vllm")
    config = get_config()

    print(f"\n[Device: {device}]")
    print(f"  Config: {config['device']}:{config['engine']}")

    try:
        from vllm import LLM, SamplingParams

        start = time.time()
        llm = LLM(model=model)
        load_time = time.time() - start

        print(f"  Backend: {llm.backend}")
        print(f"  Load time: {load_time:.2f}s")

        # Generate
        start = time.time()
        outputs = llm.generate(
            ["Hello, world!"],
            SamplingParams(max_tokens=20)
        )
        gen_time = time.time() - start

        text = outputs[0].outputs[0].text.strip().replace("\n", " ")[:50]
        print(f"  Gen time: {gen_time:.2f}s")
        print(f"  Output: {text}...")

        return True

    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 60)
    print("Device Comparison Test")
    print("=" * 60)

    devices = ["cpu"]  # Always test CPU

    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            devices.append("nvidia")
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    # Check for NPUs (would need actual SDK)
    # devices.append("rbln")    # Uncomment if Rebellions SDK installed
    # devices.append("furiosa") # Uncomment if FuriosaAI SDK installed

    print(f"Testing devices: {devices}")

    results = {}
    for device in devices:
        results[device] = test_device(device)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for device, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {device}: {status}")


if __name__ == "__main__":
    main()
