"""Schema exports for the 41-sdprovider project.

This module centralizes exports for commonly used Pydantic models so other
modules can import from src.schemas directly when convenient.

Author: Inventions4All - github:TWeb79
"""

# Health schemas
from .health import HealthResponse

# Generation schemas
from .generate import GenerateRequest, GenerateResponse, GenerateError

# Model schemas
from .model import (
    ModelInfo,
    ModelListResponse,
    ModelLoadRequest,
    ModelLoadResponse,
)

# Load schemas (for /load-model endpoint)
from .load import (
    LoadModelRequest,
    LoadModelResponse,
    LoadModelError,
)

__all__ = [
    # Health
    "HealthResponse",
    # Generation
    "GenerateRequest",
    "GenerateResponse",
    "GenerateError",
    # Model
    "ModelInfo",
    "ModelListResponse",
    "ModelLoadRequest",
    "ModelLoadResponse",
    # Load
    "LoadModelRequest",
    "LoadModelResponse",
    "LoadModelError",
]