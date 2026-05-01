"""
Health check endpoint for the Stable Diffusion API.

Author: Inventions4All - github:TWeb79
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from src.schemas.health import HealthResponse
from src.core.device import get_device_info
from src.core.pipeline import get_pipeline_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns service health status and system information.",
)
async def health_check() -> HealthResponse:
    """
    Check the health of the service.
    
    Returns:
        HealthResponse with service status, loaded model info, and device info
    """
    pipeline_manager = get_pipeline_manager()
    device_info = get_device_info()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        loaded_model=pipeline_manager.current_model,
        device=device_info.name,
        device_type=device_info.type,
        mps_available=device_info.mps_available,
        torch_num_threads=device_info.num_threads or 0,
        torch_interop_threads=device_info.interop_threads or 0,
    )