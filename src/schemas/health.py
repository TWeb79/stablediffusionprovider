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
    device: str = Field(description="Device name (CPU or Apple MPS)")
    device_type: str = Field(description="Device type (cpu/mps)")
    mps_available: bool = Field(description="Whether Apple Metal (MPS) is available")
    torch_num_threads: int = Field(description="torch.get_num_threads() value")
    torch_interop_threads: int = Field(description="torch.get_num_interop_threads() value")