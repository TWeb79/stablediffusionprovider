"""
Request/Response schemas for image generation.

Author: Inventions4All - github:TWeb79
"""

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """
    Request schema for image generation.
    
    All parameters are optional with sensible defaults.
    """
    
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text description of the desired image",
    )
    negative_prompt: str = Field(
        default="",
        max_length=1000,
        description="Things to avoid in the image",
    )
    steps: int = Field(
        default=25,
        ge=1,
        le=150,
        description="Number of inference steps",
    )
    guidance: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="Guidance scale (higher = more prompt adherence)",
    )
    width: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Image width in pixels",
    )
    height: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Image height in pixels",
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Random seed (0 = random)",
    )
    model_path: str | None = Field(
        default=None,
        description="Full path to model file (e.g., /models/model.safetensors). If not provided, uses default model.",
    )
    
    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Ensure dimensions are multiples of 8 for Stable Diffusion."""
        if v % 8 != 0:
            raise ValueError("Image dimensions must be multiples of 8")
        return v


class GenerateResponse(BaseModel):
    """Response schema for generation status."""
    
    success: bool = Field(description="Whether generation succeeded")
    model: str = Field(description="Model used for generation")
    seed: int = Field(description="Seed used for generation")
    parameters: dict = Field(description="Generation parameters used")
    generation_time_ms: float = Field(description="Time taken for generation")


class GenerateError(BaseModel):
    """Error response schema."""
    
    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Detailed error info")