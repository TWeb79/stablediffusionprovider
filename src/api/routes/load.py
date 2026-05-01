"""
Model loading endpoint for the Stable Diffusion API.

Author: Inventions4All - github:TWeb79
"""

import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ...schemas.model import LoadModelRequest
from ...core.pipeline import get_pipeline_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Model Loading"])


@router.post(
    "/load-model",
    summary="Load Model from Path",
    description="Loads a Stable Diffusion model from a specific file path.",
)
async def load_model(request: LoadModelRequest) -> dict:
    """
    Load a model from a full file path.
    
    Args:
        request: LoadModelRequest containing the model_path
        
    Returns:
        Success status with model info
        
    Raises:
        HTTPException: If model file not found or loading fails
    """
    pipeline_manager = get_pipeline_manager()
    model_path = request.model_path

    # Validate file exists using os.path.isfile
    if not os.path.isfile(model_path):
        logger.error(f"Model file not found at path: {model_path}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Model file not found at path: {model_path}"},
        )

    try:
        pipeline_manager.load_model(model_path)
        
        return {
            "success": True,
            "model_path": model_path,
            "model": pipeline_manager.current_model,
            "device": pipeline_manager.device,
        }
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to load model: {str(e)}"},
        )
