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
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """API server configuration settings."""
    
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8141, description="API server port (project 41)")
    workers: int = Field(default=1, description="Number of worker processes")
    
    class Config:
        env_prefix = "API_"


class ModelSettings(BaseSettings):
    """Model configuration settings."""
    
    directory: str = Field(default="/models", description="Path to model files")
    default_model: Optional[str] = Field(default=None, description="Default model to load")
    safety_checker: bool = Field(default=False, description="Enable safety checker")
    
    class Config:
        env_prefix = "MODEL_"
        alias_generator = lambda x: x.upper()


class DeviceSettings(BaseSettings):
    """Device configuration settings."""
    
    device: str = Field(default="cuda", description="Compute device (cuda/cpu)")
    attention_slicing: bool = Field(default=True, description="Enable attention slicing")
    cpu_offload: bool = Field(default=False, description="Enable CPU offload")
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device is either cuda or cpu."""
        if v not in ("cuda", "cpu"):
            raise ValueError("Device must be 'cuda' or 'cpu'")
        return v
    
    class Config:
        env_prefix = ""


class GenerationSettings(BaseSettings):
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
    
    class Config:
        env_prefix = "DEFAULT_"


class Settings(BaseSettings):
    """
    Main application settings container.
    
    Loads from environment variables with fallback to config.yml.
    Environment variables take precedence over config file values.
    """
    
    api: APISettings = Field(default_factory=APISettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    device: DeviceSettings = Field(default_factory=DeviceSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    
    # HuggingFace token for gated models
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
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
        
        # Build settings from YAML config
        api_config = yaml_config.get("api", {})
        model_config = yaml_config.get("model", {})
        device_config = yaml_config.get("device", {})
        generation_config = yaml_config.get("generation", {})
        
        return cls(
            api=APISettings(**api_config),
            model=ModelSettings(**model_config),
            device=DeviceSettings(**device_config),
            generation=GenerationSettings(**generation_config),
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