"""
Health check response schema.

Author: Inventions4All - github:TWeb79
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    
    status: str = Field(description="Service status (healthy/unhealthy)")
    timestamp: str = Field(description="ISO timestamp of the health check")
    loaded_model: str | None = Field(
        default=None,
        description="Currently loaded model name",
    )
    device: str = Field(description="Device name (e.g., NVIDIA GPU, CPU)")
    device_type: str = Field(description="Device type (cuda/cpu)")
    cuda_available: bool = Field(description="Whether CUDA is available")
    cuda_device_count: int = Field(description="Number of CUDA devices")