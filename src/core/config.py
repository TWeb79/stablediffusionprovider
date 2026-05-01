"""
Configuration management for Stable Diffusion API Service.

Loads configuration from environment variables and config.yml file.
Provides type-safe settings with validation and sensible defaults.

Author: Inventions4All - github:TWeb79
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class APISettings(BaseModel):
    """API server configuration settings."""
    
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8141, description="API server port (project 41)")
    workers: int = Field(default=1, description="Number of worker processes")


class ModelSettings(BaseModel):
    """Model configuration settings."""
    
    directory: str = Field(default="/models", description="Path to model files")
    default_model: Optional[str] = Field(default=None, description="Default model to load")
    safety_checker: bool = Field(default=False, description="Enable safety checker")


class DeviceSettings(BaseModel):
    """Device configuration settings."""
    
    device: str = Field(default="cuda", description="Compute device (cuda/cpu/mps/auto)")
    attention_slicing: bool = Field(default=True, description="Enable attention slicing")
    cpu_offload: bool = Field(default=False, description="Enable CPU offload")
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device is either cuda, cpu, mps, or auto."""
        if v not in ("cuda", "cpu", "mps", "auto"):
            raise ValueError("Device must be 'cuda', 'cpu', 'mps', or 'auto'")
        return v


class GenerationSettings(BaseModel):
    """Default generation parameters."""
    
    steps: int = Field(default=25, ge=1, le=150, description="Default inference steps")
    guidance: float = Field(default=7.5, ge=1.0, le=20.0, description="Default guidance scale")
    width: int = Field(default=512, ge=256, le=1024, description="Default image width")
    height: int = Field(default=512, ge=256, le=1024, description="Default image height")
    
    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Ensure dimensions are multiples of 8 for Stable Diffusion."""
        if v % 8 != 0:
            raise ValueError("Image dimensions must be multiples of 8")
        return v


class Settings(BaseModel):
    """
    Main application settings container.
    
    Loads from environment variables with fallback to config.yml.
    Environment variables take precedence over config file values.
    """
    
    # Flat env var fields for API settings
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8141, alias="API_PORT")
    api_workers: int = Field(default=1, alias="API_WORKERS")
    
    # Flat env var fields for model settings
    model_directory: str = Field(default="/models", alias="MODEL_DIR")
    model_default_model: Optional[str] = Field(default=None, alias="DEFAULT_MODEL")
    model_safety_checker: bool = Field(default=False, alias="SAFETY_CHECKER")
    
    # Flat env var fields for device settings
    device_device: str = Field(default="cuda", alias="DEVICE")
    device_attention_slicing: bool = Field(default=True, alias="ATTENTION_SLICING")
    device_cpu_offload: bool = Field(default=False, alias="CPU_OFFLOAD")
    
    # Flat env var fields for generation settings
    generation_steps: int = Field(default=25, alias="DEFAULT_STEPS")
    generation_guidance: float = Field(default=7.5, alias="DEFAULT_GUIDANCE")
    generation_width: int = Field(default=512, alias="DEFAULT_WIDTH")
    generation_height: int = Field(default=512, alias="DEFAULT_HEIGHT")
    
    # HuggingFace token for gated models
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> "Settings":
        """
        Load settings from YAML configuration file.
        
        Args:
            config_path: Path to config.yml file. Defaults to ./config/config.yml
            
        Returns:
            Settings instance with merged configuration
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yml"
        
        yaml_config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
        
        # Extract YAML values
        api_config = yaml_config.get("api", {})
        model_config = yaml_config.get("model", {})
        device_config = yaml_config.get("device", {})
        generation_config = yaml_config.get("generation", {})
        
        # Build kwargs, starting with env vars (which have aliases)
        # Then override with YAML values where env vars weren't set
        kwargs = {}
        
        # API settings - use YAML values as defaults, env vars override
        kwargs["api_host"] = os.environ.get("API_HOST", api_config.get("host", "0.0.0.0")) or "0.0.0.0"
        kwargs["api_port"] = int(os.environ.get("API_PORT", str(api_config.get("port", 8141))) or 8141)
        kwargs["api_workers"] = int(os.environ.get("API_WORKERS", str(api_config.get("workers", 1))) or 1)
        
        # Model settings
        kwargs["model_directory"] = os.environ.get("MODEL_DIR", model_config.get("directory", "/models")) or "/models"
        # Support both DEFAULT_MODEL_PATH (new) and DEFAULT_MODEL (legacy)
        kwargs["model_default_model"] = (
            os.environ.get("DEFAULT_MODEL_PATH") or 
            os.environ.get("DEFAULT_MODEL") or 
            model_config.get("default_model")
        )
        kwargs["model_safety_checker"] = os.environ.get("SAFETY_CHECKER", "").lower() in ("true", "1", "yes") if "SAFETY_CHECKER" in os.environ else model_config.get("safety_checker", False)
        
        # Device settings
        kwargs["device_device"] = os.environ.get("DEVICE", device_config.get("device", "cuda")) or "cuda"
        kwargs["device_attention_slicing"] = os.environ.get("ATTENTION_SLICING", "").lower() in ("true", "1", "yes") if "ATTENTION_SLICING" in os.environ else device_config.get("attention_slicing", True)
        kwargs["device_cpu_offload"] = os.environ.get("CPU_OFFLOAD", "").lower() in ("true", "1", "yes") if "CPU_OFFLOAD" in os.environ else device_config.get("cpu_offload", False)
        
        # Generation settings
        kwargs["generation_steps"] = int(os.environ.get("DEFAULT_STEPS", str(generation_config.get("steps", 25))) or 25)
        kwargs["generation_guidance"] = float(os.environ.get("DEFAULT_GUIDANCE", str(generation_config.get("guidance", 7.5))) or 7.5)
        kwargs["generation_width"] = int(os.environ.get("DEFAULT_WIDTH", str(generation_config.get("width", 512))) or 512)
        kwargs["generation_height"] = int(os.environ.get("DEFAULT_HEIGHT", str(generation_config.get("height", 512))) or 512)
        
        # HuggingFace token
        kwargs["hf_token"] = os.environ.get("HF_TOKEN")
        
        # Logging
        kwargs["log_level"] = os.environ.get("LOG_LEVEL", "INFO")
        
        return cls(**kwargs)
    
    @property
    def api(self) -> APISettings:
        """Get API settings as APISettings object."""
        return APISettings(
            host=self.api_host,
            port=self.api_port,
            workers=self.api_workers,
        )
    
    @property
    def model(self) -> ModelSettings:
        """Get model settings as ModelSettings object."""
        return ModelSettings(
            directory=self.model_directory,
            default_model=self.model_default_model,
            safety_checker=self.model_safety_checker,
        )
    
    @property
    def device(self) -> DeviceSettings:
        """Get device settings as DeviceSettings object."""
        return DeviceSettings(
            device=self.device_device,
            attention_slicing=self.device_attention_slicing,
            cpu_offload=self.device_cpu_offload,
        )
    
    @property
    def generation(self) -> GenerationSettings:
        """Get generation settings as GenerationSettings object."""
        return GenerationSettings(
            steps=self.generation_steps,
            guidance=self.generation_guidance,
            width=self.generation_width,
            height=self.generation_height,
        )


# Global settings instance - lazy loaded
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Loads settings on first access, then caches for subsequent calls.
    
    Returns:
        Settings instance with all configuration values
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml()
    return _settings


def reload_settings() -> Settings:
    """
    Force reload settings from configuration sources.
    
    Useful for testing or dynamic configuration updates.
    
    Returns:
        Fresh Settings instance
    """
    global _settings
    _settings = Settings.from_yaml()
    return _settings