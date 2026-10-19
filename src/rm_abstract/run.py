"""
RM Abstract Launcher

Run existing scripts with automatic backend switching without code modification.

Usage:
    # Run with auto device selection
    python -m rm_abstract.run your_script.py

    # Run with specific device
    python -m rm_abstract.run --device=rbln your_script.py

    # Run with specific engine
    python -m rm_abstract.run --device=nvidia --engine=vllm your_script.py

    # Disable specific hooks
    python -m rm_abstract.run --no-hook-vllm your_script.py

Examples:
    python -m rm_abstract.run inference.py --device=auto
    python -m rm_abstract.run --device=rbln serve.py --port 8000
"""

import argparse
import os
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run Python scripts with RM Abstract backend switching",
        usage="python -m rm_abstract.run [options] script.py [script_args...]",
    )

    parser.add_argument(
        "script",
        help="Python script to run",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="auto",
        choices=["auto", "cpu", "nvidia", "gpu", "rbln", "furiosa"],
        help="Device selection (default: auto)",
    )
    parser.add_argument(
        "--engine",
        "-e",
        default="vllm",
        choices=["vllm", "triton", "torchserve"],
        help="Serving engine (default: vllm)",
    )
    parser.add_argument(
        "--no-hook-vllm",
        action="store_true",
        help="Disable vLLM import hooking",
    )
    parser.add_argument(
        "--no-hook-transformers",
        action="store_true",
        help="Disable transformers import hooking",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Parse known args, rest goes to script
    args, script_args = parser.parse_known_args()

    # Set environment variables before importing anything
    os.environ["RM_DEVICE"] = args.device
    os.environ["RM_ENGINE"] = args.engine

    if args.no_hook_vllm:
        os.environ["RM_HOOK_VLLM"] = "0"
    if args.no_hook_transformers:
        os.environ["RM_HOOK_TRANSFORMERS"] = "0"

    # Setup logging
    if args.verbose:
        import logging

        logging.basicConfig(
            level=logging.DEBUG,
            format="[RM Abstract] %(levelname)s: %(message)s",
        )

    # Enable hooks
    import rm_abstract.auto  # noqa: F401

    # Prepare sys.argv for the target script
    sys.argv = [args.script] + script_args

    # Run the script
    print(
        f"[RM Abstract] Running {args.script} with device={args.device}, engine={args.engine}"
    )

    try:
        runpy.run_path(args.script, run_name="__main__")
    except FileNotFoundError:
        print(f"Error: Script not found: {args.script}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running script: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
