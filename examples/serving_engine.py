"""
Serving Engine Example

Demonstrates how to use serving engines (vLLM, Triton, TorchServe).
"""

from rm_abstract.engines import (
    VLLMEngine,
    TritonEngine,
    TorchServeEngine,
    ServingConfig,
    ServingEngineType,
    DeviceTarget,
)


def demo_vllm():
    """Demo vLLM serving engine"""
    print("=" * 50)
    print("vLLM Engine Demo")
    print("=" * 50)

    if not VLLMEngine.is_available():
        print("vLLM not available (install with: pip install vllm)")
        return

    config = ServingConfig(
        engine=ServingEngineType.VLLM,
        device=DeviceTarget.GPU,
        model_name="gpt2",
        port=8000,
    )

    engine = VLLMEngine(config)
    print(f"Engine: {engine.name}")
    print(f"Supported devices: {engine.supported_devices()}")

    # Note: Actual model loading requires GPU
    # engine.load_model("gpt2")
    # output = engine.infer("Hello, world!")


def demo_triton():
    """Demo Triton serving engine"""
    print("\n" + "=" * 50)
    print("Triton Engine Demo")
    print("=" * 50)

    if not TritonEngine.is_available():
        print("Triton not available (requires Docker)")
        return

    config = ServingConfig(
        engine=ServingEngineType.TRITON,
        model_name="model",
        port=8000,
    )

    engine = TritonEngine(config)
    print(f"Engine: {engine.name}")
    print(f"Supported devices: {engine.supported_devices()}")


def demo_torchserve():
    """Demo TorchServe engine"""
    print("\n" + "=" * 50)
    print("TorchServe Engine Demo")
    print("=" * 50)

    if not TorchServeEngine.is_available():
        print("TorchServe not available (install with: pip install torchserve)")
        return

    config = ServingConfig(
        engine=ServingEngineType.TORCHSERVE,
        model_name="model",
        port=8080,
    )

    engine = TorchServeEngine(config)
    print(f"Engine: {engine.name}")
    print(f"Supported devices: {engine.supported_devices()}")


def main():
    print("Serving Engine Examples")
    print("=" * 50)
    print()

    demo_vllm()
    demo_triton()
    demo_torchserve()


if __name__ == "__main__":
    main()
