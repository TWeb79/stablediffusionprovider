"""
Model information schemas.

Author: Inventions4All - github:TWeb79
"""

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about a single model file."""
    
    name: str = Field(description="Model filename")
    path: str = Field(description="Full path to model file")
    size_bytes: int = Field(description="File size in bytes")
    size_mb: float = Field(description="File size in megabytes")
    format: str = Field(description="File format (safetensors/ckpt)")


class ModelListResponse(BaseModel):
    """Response schema for model list endpoint."""
    
    models: list[ModelInfo] = Field(description="List of available models")
    count: int = Field(description="Total number of models")
    current_model: str | None = Field(
        default=None,
        description="Currently loaded model name",
    )


class ModelLoadRequest(BaseModel):
    """Request to load a specific model."""
    
    model_name: str = Field(
        ...,
        min_length=1,
        description="Name of the model to load",
    )


class ModelLoadResponse(BaseModel):
    """Response after loading a model."""
    
    success: bool = Field(description="Whether loading succeeded")
    model: str = Field(description="Model that was loaded")
    device: str = Field(description="Device the model is on")
    memory_optimizations: dict = Field(description="Applied memory optimizations")