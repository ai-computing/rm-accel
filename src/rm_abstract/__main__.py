"""
Entry point for python -m rm_abstract

Supports:
    python -m rm_abstract run script.py --device=auto
    python -m rm_abstract info
    python -m rm_abstract list-devices
"""

import sys
import argparse


def cmd_run(args):
    """Run a script with hooks enabled."""
    from rm_abstract.run import main
    sys.argv = ["rm_abstract.run"] + args
    main()


def cmd_info():
    """Show RM Abstract information."""
    import rm_abstract

    print("RM Abstract Layer")
    print("=" * 40)
    print(f"Version: {getattr(rm_abstract, '__version__', '0.1.0')}")
    print()

    # Show available resources
    print("Available Resources:")
    resources = rm_abstract.get_available_resources()
    for name, available in resources.items():
        status = "available" if available else "not available"
        print(f"  {name}: {status}")
    print()


def cmd_list_devices():
    """List available devices."""
    from rm_abstract import HardwareDetector

    detector = HardwareDetector()
    print(detector.summary())


def main():
    parser = argparse.ArgumentParser(
        prog="rm_abstract",
        description="RM Abstract Layer - Heterogeneous Accelerator Abstraction",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run a script with hooks enabled")
    run_parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Script and arguments")

    # info command
    subparsers.add_parser("info", help="Show RM Abstract information")

    # list-devices command
    subparsers.add_parser("list-devices", help="List available devices")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args.script_args)
    elif args.command == "info":
        cmd_info()
    elif args.command == "list-devices":
        cmd_list_devices()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
