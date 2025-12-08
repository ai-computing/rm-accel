"""
Device Switching Example

Demonstrates how to switch devices at runtime.
"""

import rm_abstract


def main():
    # Initialize with CPU
    print("=" * 50)
    print("Starting with CPU...")
    print("=" * 50)

    rm_abstract.init(device="cpu", verbose=True)
    print(f"Current device: {rm_abstract.get_device_info()}")

    # Switch to GPU (if available)
    print("\n" + "=" * 50)
    print("Switching to GPU...")
    print("=" * 50)

    try:
        rm_abstract.switch_device("nvidia")
        print(f"Current device: {rm_abstract.get_device_info()}")
    except Exception as e:
        print(f"GPU not available: {e}")

    # Switch to NPU (if available)
    print("\n" + "=" * 50)
    print("Switching to NPU...")
    print("=" * 50)

    try:
        rm_abstract.switch_device("rbln")
        print(f"Current device: {rm_abstract.get_device_info()}")
    except Exception as e:
        print(f"NPU not available: {e}")


if __name__ == "__main__":
    main()
