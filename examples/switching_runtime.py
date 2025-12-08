"""
Runtime Device Switching Example

Demonstrates switching between devices at runtime.
Note: Each switch requires reloading the model.

Usage:
    python examples/switching_runtime.py
"""

import sys


def create_llm(device: str, model: str = "gpt2"):
    """Create LLM with specified device."""
    # Must clean modules to switch device
    for mod in list(sys.modules.keys()):
        if "rm_abstract" in mod or mod == "vllm":
            del sys.modules[mod]

    from rm_abstract.auto import use
    use(device=device, engine="vllm")

    from vllm import LLM
    return LLM(model=model)


def main():
    print("=" * 60)
    print("Runtime Device Switching Example")
    print("=" * 60)

    # Start with CPU
    print("\n[1] Creating LLM on CPU...")
    llm_cpu = create_llm("cpu")
    print(f"    Backend: {llm_cpu.backend}")

    from vllm import SamplingParams
    params = SamplingParams(max_tokens=20)

    outputs = llm_cpu.generate(["CPU says:"], params)
    print(f"    Output: {outputs[0].outputs[0].text.strip()[:40]}...")

    # Note: To switch to GPU, you would do:
    # print("\n[2] Switching to GPU...")
    # llm_gpu = create_llm("nvidia")
    # outputs = llm_gpu.generate(["GPU says:"], params)

    # For NPU:
    # print("\n[3] Switching to NPU...")
    # llm_npu = create_llm("rbln")
    # outputs = llm_npu.generate(["NPU says:"], params)

    print("\n" + "=" * 60)
    print("To switch devices, call create_llm() with different device")
    print("  - create_llm('cpu')     -> CPU backend")
    print("  - create_llm('nvidia')  -> GPU backend")
    print("  - create_llm('rbln')    -> Rebellions NPU")
    print("  - create_llm('furiosa') -> FuriosaAI NPU")
    print("=" * 60)


if __name__ == "__main__":
    main()
