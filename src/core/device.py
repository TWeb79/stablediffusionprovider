"""
Device detection and management for Stable Diffusion API Service.

Handles CUDA/CPU device detection with automatic fallback.
Provides memory optimization settings based on available hardware.

Author: Inventions4All - github:TWeb79
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """
    Information about the compute device being used.
    
    Attributes:
        name: Human-readable device name
        type: Device type (cuda/cpu)
        memory_total: Total VRAM/RAM in bytes (None if unknown)
        memory_available: Available memory in bytes (None if unknown)
        cuda_available: Whether CUDA is available
        cuda_device_count: Number of available CUDA devices
    """
    
    name: str
    type: str
    memory_total: Optional[int] = None
    memory_available: Optional[int] = None
    cuda_available: bool = False
    cuda_device_count: int = 0


def detect_device(preferred_device: Optional[str] = None) -> str:
    """
    Detect and return the best available compute device.
    
    Args:
        preferred_device: User-specified device preference (cuda/cpu/mps/auto)
        
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if preferred_device and preferred_device in ("cuda", "cpu", "mps"):
        if preferred_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        if preferred_device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return preferred_device
    
    # Auto-detect: prefer CUDA if available, then MPS, then CPU
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available: {device_name}")
        return "cuda"
    
    # Check for Apple Silicon MPS support
    if torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) available")
        return "mps"
    
    logger.info("Using CPU")
    return "cpu"


def get_device_info(device: Optional[str] = None) -> DeviceInfo:
    """
    Get detailed information about the compute device.
    
    Args:
        device: Device to query (defaults to auto-detect)
        
    Returns:
        DeviceInfo with hardware details
    """
    if device is None:
        device = detect_device()
    
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    
    if device == "cuda" and cuda_available:
        # Get CUDA device properties
        props = torch.cuda.get_device_properties(0)
        memory_total = props.total_memory
        memory_available = torch.cuda.mem_get_info()[0] if hasattr(torch.cuda, 'mem_get_info') else None
        
        return DeviceInfo(
            name=torch.cuda.get_device_name(0),
            type="cuda",
            memory_total=memory_total,
            memory_available=memory_available,
            cuda_available=True,
            cuda_device_count=cuda_device_count,
        )
    
    # CPU info
    return DeviceInfo(
        name="CPU",
        type="cpu",
        memory_total=None,
        memory_available=None,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
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


def optimize_for_device(device: str, attention_slicing: bool = True, cpu_offload: bool = False) -> dict:
    """
    Get optimization settings for the specified device.
    
    Args:
        device: Target device (cuda/cpu/mps)
        attention_slicing: Enable attention slicing for memory savings
        cpu_offload: Enable CPU offload for additional memory savings
        
    Returns:
        Dictionary of optimization flags
    """
    # For GPU devices (cuda/mps), use user settings; for CPU, always enable optimizations
    is_gpu = device in ("cuda", "mps")
    
    settings = {
        "device": device,
        "attention_slicing": attention_slicing if is_gpu else True,
        "cpu_offload": cpu_offload if is_gpu else True,
        "enable_vae_slicing": attention_slicing if is_gpu else True,
        "enable_sequential_cpu_offload": cpu_offload if is_gpu else True,
    }
    
    logger.info(f"Device optimization settings: {settings}")
    return settings
