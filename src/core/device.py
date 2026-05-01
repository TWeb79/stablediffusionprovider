"""
Device detection and runtime configuration utilities for the Stable Diffusion API.

Docker deployments must run on CPU. Local executions may optionally leverage
Apple Silicon's Metal Performance Shaders (MPS) backend when available.

Author: Inventions4All - github:TWeb79
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about the compute device being used."""
    
    name: str
    type: str
    memory_total: Optional[int] = None
    memory_available: Optional[int] = None
    mps_available: bool = False
    num_threads: Optional[int] = None
    interop_threads: Optional[int] = None


def detect_device(preferred_device: Optional[str] = None) -> str:
    """Detect and return the best available compute device (cpu/mps)."""
    allowed = {"cpu", "mps", "auto", None}
    if preferred_device not in allowed:
        logger.warning("Unsupported device '%s', forcing CPU", preferred_device)
        return "cpu"

    if preferred_device == "mps":
        if torch.backends.mps.is_available():
            logger.info("Using Apple MPS backend")
            return "mps"
        logger.warning("MPS requested but not available, falling back to CPU")
        return "cpu"

    if preferred_device == "cpu":
        return "cpu"

    # Auto mode: prefer MPS when available, else CPU
    if torch.backends.mps.is_available():
        logger.info("Auto-detected MPS backend")
        return "mps"

    logger.info("Auto-detected CPU backend")
    return "cpu"


def get_device_info(device: Optional[str] = None) -> DeviceInfo:
    """Get detailed information about the compute device."""
    if device is None:
        device = detect_device()

    mps_available = torch.backends.mps.is_available()

    if device == "mps" and mps_available:
        return DeviceInfo(
            name="Apple Metal (MPS)",
            type="mps",
            memory_total=None,
            memory_available=None,
            mps_available=True,
            num_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
        )

    return DeviceInfo(
        name="CPU",
        type="cpu",
        memory_total=None,
        memory_available=None,
        mps_available=mps_available,
        num_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
    )


def get_memory_requirements(width: int, height: int, model_size_gb: float = 4.0) -> dict:
    """
    Estimate memory requirements for image generation.
    
    Approximate VRAM requirements based on image dimensions and model.
    Actual requirements may vary based on model complexity and settings.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        model_size_gb: Model size in GB (default: 4GB for SD 1.5)
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Calculate pixel count
    pixels = width * height
    
    # Base model memory
    model_memory = model_size_gb
    
    # Inference memory scales roughly with pixel count
    # Higher resolution = more memory for attention maps
    inference_memory = (pixels / (512 * 512)) * 2.0
    
    # Total estimated VRAM
    total = model_memory + inference_memory
    
    return {
        "model_gb": model_memory,
        "inference_gb": round(inference_memory, 2),
        "total_gb": round(total, 2),
        "recommended_gb": round(total * 1.2, 2),  # 20% buffer
    }


def optimize_for_device(device: str, attention_slicing: bool = True, cpu_offload: bool = True) -> dict:
    """Return optimization settings for the specified device."""
    is_mps = device == "mps"
    settings = {
        "device": device,
        "attention_slicing": True if not is_mps else attention_slicing,
        "cpu_offload": cpu_offload if device == "cpu" else False,
        "enable_vae_slicing": True,
        "enable_sequential_cpu_offload": cpu_offload if device == "cpu" else False,
    }
    logger.info("Device optimization settings: %s", settings)
    return settings


def configure_torch_runtime(
    num_threads: Optional[int] = None,
    interop_threads: Optional[int] = None,
    omp_threads: Optional[int] = None,
    mkl_threads: Optional[int] = None,
) -> None:
    """Configure torch and environment thread settings for optimal CPU usage."""
    if num_threads:
        torch.set_num_threads(num_threads)
        logger.info("Set torch.set_num_threads(%s)", num_threads)
    if interop_threads:
        torch.set_num_interop_threads(interop_threads)
        logger.info("Set torch.set_num_interop_threads(%s)", interop_threads)
    if omp_threads:
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        logger.info("Set OMP_NUM_THREADS=%s", omp_threads)
    if mkl_threads:
        os.environ["MKL_NUM_THREADS"] = str(mkl_threads)
        logger.info("Set MKL_NUM_THREADS=%s", mkl_threads)
