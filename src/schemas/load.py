"""
Request/Response schemas for model loading.

Author: Inventions4All - github:TWeb79
"""

from pydantic import BaseModel, Field


class LoadModelRequest(BaseModel):
    """
    Request to load a model by full file path.
    
    This schema is used by the /load-model endpoint to load
    a Stable Diffusion model from a specific file path.
    """
    
    model_path: str = Field(
        ...,
        min_length=1,
        description="Full path to the model file (e.g., /models/model.safetensors)",
    )


class LoadModelResponse(BaseModel):
    """
    Response after loading a model from a full file path.
    
    Contains the result of the model loading operation including
    success status, model info, and device information.
    """
    
    success: bool = Field(description="Whether model loading succeeded")
    model_path: str = Field(description="Full path that was requested for loading")
    model: str = Field(description="Identifier of the model now active in the pipeline")
    device: str = Field(description="Device where the model is loaded (cpu/mps)")


class LoadModelError(BaseModel):
    """Error response schema for model loading failures."""
    
    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")