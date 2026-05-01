#!/usr/bin/env python3
"""
Stable Diffusion API - Local Runner Entry Point.

This script provides a convenient way to run the Stable Diffusion API service
on macOS with Apple Metal (MPS) acceleration when available, or CPU fallback.

Author: Inventions4All - github:TWeb79
"""

import argparse
import logging
import os
import sys

import torch

from src.core.config import get_settings, reload_settings
from src.core.device import configure_torch_runtime, detect_device, get_device_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion API locally with MPS/CPU support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with auto-detected device (MPS if available, else CPU)
  python run_local.py

  # Run with specific model directory
  python run_local.py --model-dir ./models

  # Run on CPU only
  python run_local.py --device cpu

  # Run with custom port and thread settings
  python run_local.py --port 8141 --threads 8 --interop-threads 4
        """,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8141,
        help="API server port (default: 8141)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to model files (default: from env or /models)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Compute device: auto (MPS if available), cpu, or mps (default: auto)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of torch threads (default: auto-detect)",
    )
    parser.add_argument(
        "--interop-threads",
        type=int,
        default=None,
        help="Number of torch interop threads (default: auto-detect)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> str:
    """
    Configure environment for local execution.

    Detects MPS availability, configures threading, and returns the device to use.

    Args:
        args: Parsed command line arguments

    Returns:
        Device string ("cpu" or "mps")
    """
    # Set environment variables before importing app
    if args.model_dir:
        os.environ["MODEL_DIR"] = args.model_dir

    if args.device != "auto":
        os.environ["DEVICE"] = args.device

    if args.threads:
        os.environ["TORCH_NUM_THREADS"] = str(args.threads)

    if args.interop_threads:
        os.environ["TORCH_INTEROP_THREADS"] = str(args.interop_threads)

    os.environ["LOG_LEVEL"] = args.log_level

    # Detect device
    device = detect_device(args.device)

    # Configure threading based on device and system
    num_cores = os.cpu_count() or 4

    if args.threads is None:
        # Auto-configure threads: use half of available cores for torch
        threads = max(2, num_cores // 2)
    else:
        threads = args.threads

    if args.interop_threads is None:
        interop_threads = max(2, num_cores // 4)
    else:
        interop_threads = args.interop_threads

    configure_torch_runtime(
        num_threads=threads,
        interop_threads=interop_threads,
        omp_threads=threads,
        mkl_threads=threads,
    )

    return device


def print_startup_info(device: str, args: argparse.Namespace) -> None:
    """Print startup information."""
    device_info = get_device_info(device)
    mps_available = torch.backends.mps.is_available()

    print("\n" + "=" * 60)
    print("  Stable Diffusion API - Local Runner")
    print("=" * 60)
    print(f"  Device:        {device_info.name}")
    print(f"  Device Type:  {device_info.type}")
    print(f"  MPS Available: {mps_available}")
    print(f"  Torch Threads: {device_info.num_threads}")
    print(f"  Interop Threads: {device_info.interop_threads}")
    print("-" * 60)
    print(f"  Host:          {args.host}")
    print(f"  Port:          {args.port}")
    print(f"  Model Dir:     {args.model_dir or 'from config'}")
    print("=" * 60 + "\n")


def run_server(args: argparse.Namespace) -> None:
    """Run the FastAPI server."""
    import uvicorn

    # Reload settings to pick up environment changes
    settings = reload_settings()

    # Override with CLI args
    settings.api_host = args.host
    settings.api_port = args.port
    settings.log_level = args.log_level

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure environment and detect device
    device = configure_environment(args)

    # Print startup information
    print_startup_info(device, args)

    # Run the server
    run_server(args)


if __name__ == "__main__":
    main()