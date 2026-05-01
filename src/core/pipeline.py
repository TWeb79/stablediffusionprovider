"""
Stable Diffusion Pipeline Manager.

Handles model loading, caching, and image generation.
Supports lazy loading, model switching, and memory optimization.

Author: Inventions4All - github:TWeb79
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image

from .device import detect_device, optimize_for_device

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Manages Stable Diffusion pipeline lifecycle and generation.
    
    Handles lazy loading of models, memory optimization, and
    provides a clean interface for image generation.
    """
    
    def __init__(
        self,
        model_dir: str = "/models",
        device: Optional[str] = None,
        safety_checker: bool = False,
        attention_slicing: bool = True,
        cpu_offload: bool = False,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the pipeline manager.
        
        Args:
            model_dir: Directory containing model files
            device: Compute device (cuda/cpu/auto)
            safety_checker: Enable safety checker
            attention_slicing: Enable attention slicing for memory savings
            cpu_offload: Enable CPU offload for additional memory savings
            hf_token: HuggingFace token for gated models
        """
        self.model_dir = Path(model_dir)
        self.device = detect_device(device)
        self.safety_checker = safety_checker
        self.hf_token = hf_token
        
        # Optimization settings
        self.attention_slicing = attention_slicing
        self.cpu_offload = cpu_offload
        
        # Pipeline state
        self._pipeline: Optional[StableDiffusionPipeline] = None
        self._current_model: Optional[str] = None
        self._model_cache: dict[str, StableDiffusionPipeline] = {}
        
        logger.info(f"PipelineManager initialized with device: {self.device}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._pipeline is not None
    
    @property
    def current_model(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        return self._current_model
    
    def discover_models(self) -> list[dict[str, Any]]:
        """
        Discover available models in the model directory.
        
        Scans for .safetensors and .ckpt files.
        
        Returns:
            List of model info dictionaries with name, path, and size
        """
        models = []
        
        if not self.model_dir.exists():
            logger.warning(f"Model directory does not exist: {self.model_dir}")
            return models
        
        for file_path in self.model_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in (".safetensors", ".ckpt"):
                stat = file_path.stat()
                models.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "format": file_path.suffix.lower().lstrip("."),
                })
        
        # Sort by name
        models.sort(key=lambda x: x["name"])
        logger.info(f"Discovered {len(models)} models in {self.model_dir}")
        return models
    
    def load_model(self, model_path: Optional[str] = None) -> StableDiffusionPipeline:
        """
        Load a Stable Diffusion model into the pipeline.

        Args:
            model_path: Full path to model file (e.g., "/models/model.safetensors")
                or model name (e.g., "model.safetensors"). If None, uses default
                model from settings or auto-detects.

        Returns:
            Loaded StableDiffusionPipeline instance

        Raises:
            FileNotFoundError: If specified model file doesn't exist
            ValueError: If no model path provided and no default found
        """
        # Resolve the model identifier to an actual file path
        path = self._resolve_model_path(model_path)
        model_key = model_path if model_path else str(path)

        # Validate file exists (should be guaranteed by _resolve_model_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Model file not found at path: {model_key}")

        # If same model already loaded, return it
        if self._current_model == model_key and self._pipeline:
            logger.info(f"Model {model_key} already loaded")
            return self._pipeline

        # Check cache first
        if model_key in self._model_cache:
            logger.info(f"Loading {model_key} from cache")
            self._pipeline = self._model_cache[model_key]
            self._current_model = model_key
            return self._pipeline

        logger.info(f"Loading model from: {path}")

        # Load the pipeline using from_single_file
        self._pipeline = StableDiffusionPipeline.from_single_file(
            str(path),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None if not self.safety_checker else None,
            requires_safety_checker=self.safety_checker,
            token=self.hf_token,
        )

        # Apply optimizations
        self._apply_optimizations()

        self._current_model = model_key

        # Cache the pipeline
        self._model_cache[model_key] = self._pipeline

        logger.info(f"Model loaded: {self._current_model}")
        return self._pipeline

    def _resolve_model_path(self, model_identifier: Optional[str] = None) -> Path:
        """
        Resolve a model identifier (path or name) to an absolute file path.

        Args:
            model_identifier: Either a full path to a model file, a model name,
                or None to use default/auto-detect.

        Returns:
            Absolute Path to the model file

        Raises:
            FileNotFoundError: If model cannot be found
        """
        if model_identifier:
            # Try as direct path first
            p = Path(model_identifier)
            if p.is_absolute() or p.parent != Path('.'):
                # Contains directory component or absolute - treat as direct path
                if p.exists() and p.is_file():
                    return p.resolve()
                raise FileNotFoundError(f"Model file not found: {model_identifier}")
            else:
                # Treat as model name, search in model_dir
                for ext in (".safetensors", ".ckpt"):
                    path = self.model_dir / f"{model_identifier}{ext}"
                    if path.exists() and path.is_file():
                        return path.resolve()
                    path = self.model_dir / model_identifier
                    if path.exists() and path.is_file():
                        return path.resolve()
                raise FileNotFoundError(f"Model not found: {model_identifier}")

        # No identifier provided: use default model from settings or auto-detect
        from .config import get_settings
        settings = get_settings()
        if settings.model.default_model:
            default_path = Path(settings.model.default_model)
            if default_path.exists() and default_path.is_file():
                return default_path.resolve()
            # If default_model is a name, try in model_dir
            for ext in (".safetensors", ".ckpt"):
                path = self.model_dir / f"{settings.model.default_model}{ext}"
                if path.exists() and path.is_file():
                    return path.resolve()
            raise FileNotFoundError(f"Default model not found: {settings.model.default_model}")

        # Auto-detect first available model
        for ext in (".safetensors", ".ckpt"):
            for path in self.model_dir.rglob(f"*{ext}"):
                if path.is_file():
                    return path.resolve()

        raise ValueError(f"No models found in {self.model_dir}")
    
    def _apply_optimizations(self) -> None:
        """Apply memory optimizations to the loaded pipeline."""
        if self._pipeline is None:
            return
        
        # Move to device
        self._pipeline = self._pipeline.to(self.device)
        
        # Attention slicing - reduces memory at slight speed cost
        if self.attention_slicing:
            self._pipeline.enable_attention_slicing()
            logger.info("Enabled attention slicing")
        
        # VAE slicing for large images
        self._pipeline.enable_vae_slicing()
        logger.info("Enabled VAE slicing")
        
        # CPU offload - significant memory savings, slower
        if self.cpu_offload:
            self._pipeline.enable_sequential_cpu_offload()
            logger.info("Enabled sequential CPU offload")
        
        # Set scheduler to DPM++ for quality
        self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipeline.scheduler.config
        )
        logger.info("Configured DPM++ scheduler")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 25,
        guidance: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: Things to avoid in the image
            steps: Number of inference steps
            guidance: Guidance scale (higher = more prompt adherence)
            width: Image width in pixels
            height: Image height in pixels
            seed: Random seed (0 = random)
            
        Returns:
            Generated PIL Image
            
        Raises:
            RuntimeError: If no model is loaded
        """
        if self._pipeline is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Set random seed for reproducibility
        generator = torch.Generator(device=self.device)
        if seed == 0:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        logger.info(f"Generating image: {width}x{height}, steps={steps}, guidance={guidance}")
        
        result = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )
        
        return result.images[0]
    
    def unload(self) -> None:
        """Unload the current model and free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._current_model = None
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded and memory freed")
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        self._model_cache.clear()
        self.unload()
        logger.info("Model cache cleared")


# Global pipeline instance
_pipeline_manager: Optional[PipelineManager] = None


def get_pipeline_manager() -> PipelineManager:
    """
    Get the global pipeline manager instance.
    
    Returns:
        PipelineManager instance
    """
    global _pipeline_manager
    if _pipeline_manager is None:
        from .config import get_settings
        settings = get_settings()
        
        _pipeline_manager = PipelineManager(
            model_dir=settings.model.directory,
            device=settings.device.device,
            safety_checker=settings.model.safety_checker,
            attention_slicing=settings.device.attention_slicing,
            cpu_offload=settings.device.cpu_offload,
            hf_token=settings.hf_token,
        )
    
    return _pipeline_manager