"""
Basic Usage Example

Demonstrates basic usage of RM Abstract Layer.
"""

import rm_abstract


def main():
    # 1. Check available resources
    print("Available resources:")
    for name, available in rm_abstract.get_available_resources().items():
        status = "OK" if available else "Not available"
        print(f"  {name}: {status}")

    # 2. Initialize with auto-selection (NPU > GPU > CPU)
    rm_abstract.init(device="auto", verbose=True)

    # 3. Get device info
    print("\nDevice Info:")
    info = rm_abstract.get_device_info()
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
