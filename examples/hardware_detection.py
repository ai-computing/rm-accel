"""
Hardware Detection Example

Demonstrates how to use HardwareDetector to discover available hardware.
"""

from rm_abstract import HardwareDetector


def main():
    detector = HardwareDetector()

    # Get summary of all hardware
    print("=" * 50)
    print("Hardware Detection")
    print("=" * 50)
    print(detector.summary())

    # Check specific vendors
    print("\n" + "=" * 50)
    print("Vendor Availability")
    print("=" * 50)

    vendors = ["nvidia", "rebellions", "furiosa", "cpu"]
    for vendor in vendors:
        available = detector.is_available(vendor)
        status = "Available" if available else "Not available"
        print(f"  {vendor}: {status}")

    # Get best available hardware
    print("\n" + "=" * 50)
    print("Best Available Hardware")
    print("=" * 50)

    best = detector.get_best_available()
    if best:
        print(f"  Type: {best.hardware_type.value}")
        print(f"  Vendor: {best.vendor.value}")
        print(f"  Name: {best.name}")
        print(f"  Device ID: {best.device_id}")
    else:
        print("  No hardware detected")


if __name__ == "__main__":
    main()
