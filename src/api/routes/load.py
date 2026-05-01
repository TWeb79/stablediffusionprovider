"""
Model loading endpoint for the Stable Diffusion API.

Author: Inventions4All - github:TWeb79
"""

import logging
import os

from fastapi import APIRouter, HTTPException

from src.schemas.load import LoadModelRequest, LoadModelResponse
from src.core.pipeline import get_pipeline_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Model Loading"])


@router.post(
    "/load-model",
    response_model=LoadModelResponse,
    summary="Load Model from Path",
    description="Loads a Stable Diffusion model from a specific file path.",
)
async def load_model(request: LoadModelRequest) -> LoadModelResponse:
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
        # Try treating as model name in /models directory
        model_dir = "/models"
        resolved_path = None
        
        for ext in (".safetensors", ".ckpt"):
            # Try with extension appended
            candidate = os.path.join(model_dir, f"{model_path}{ext}")
            if os.path.isfile(candidate):
                resolved_path = candidate
                break
            # Try if already has extension
            if model_path.endswith(ext):
                candidate = os.path.join(model_dir, os.path.basename(model_path))
                if os.path.isfile(candidate):
                    resolved_path = candidate
                    break
        
        if resolved_path:
            model_path = resolved_path
            logger.info(f"Resolved model name to path: {model_path}")
        else:
            logger.error(f"Model file not found at path: {model_path}")
            raise HTTPException(
                status_code=400,
                detail={"error": f"Model file not found at path: {model_path}"},
            )

    try:
        pipeline_manager.load_model(model_path)
        
        if pipeline_manager.current_model is None:
            raise RuntimeError("Pipeline manager did not set current_model after load")
        
        return LoadModelResponse(
            success=True,
            model_path=model_path,
            model=pipeline_manager.current_model,
            device=str(pipeline_manager.device),
        )
    except FileNotFoundError as exc:
        logger.error(f"Model resolution failed for {model_path}: {exc}")
        raise HTTPException(
            status_code=404,
            detail={"error": str(exc)},
        ) from exc
    except RuntimeError as exc:
        logger.error(f"Invalid pipeline state after loading {model_path}: {exc}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc)},
        ) from exc
    except Exception as exc:
        logger.error(f"Failed to load model from {model_path}: {exc}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to load model: {str(exc)}"},
        ) from exc