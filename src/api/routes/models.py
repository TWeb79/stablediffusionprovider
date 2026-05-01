"""
Models endpoint for the Stable Diffusion API.

Author: Inventions4All - github:TWeb79
"""

import logging

from fastapi import APIRouter, HTTPException

from ...schemas.model import (
    ModelInfo,
    ModelListResponse,
    ModelLoadResponse,
)
from ...core.pipeline import get_pipeline_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List Available Models",
    description="Returns a list of all available Stable Diffusion models.",
)
async def list_models() -> ModelListResponse:
    """
    List all available models in the model directory.
    
    Returns:
        ModelListResponse with model list and current model info
    """
    pipeline_manager = get_pipeline_manager()
    models = pipeline_manager.discover_models()
    
    return ModelListResponse(
        models=[ModelInfo(**m) for m in models],
        count=len(models),
        current_model=pipeline_manager.current_model,
    )


@router.post(
    "/{model_name}/load",
    response_model=ModelLoadResponse,
    summary="Load a Model",
    description="Loads a specific model into memory.",
)
async def load_model(model_name: str) -> ModelLoadResponse:
    """
    Load a specific model by name.
    
    Args:
        model_name: Name of the model file to load
        
    Returns:
        ModelLoadResponse with loading status and device info
        
    Raises:
        HTTPException: If model not found or loading fails
    """
    pipeline_manager = get_pipeline_manager()
    
    try:
        pipeline_manager.load_model(model_name)
        
        return ModelLoadResponse(
            success=True,
            model=pipeline_manager.current_model or model_name,
            device=pipeline_manager.device,
            memory_optimizations={
                "attention_slicing": pipeline_manager.attention_slicing,
                "cpu_offload": pipeline_manager.cpu_offload,
            },
        )
    except FileNotFoundError as e:
        logger.error(f"Model not found: {model_name}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post(
    "/unload",
    summary="Unload Current Model",
    description="Unloads the currently loaded model and frees memory.",
)
async def unload_model() -> dict:
    """
    Unload the current model from memory.
    
    Returns:
        Status message
    """
    pipeline_manager = get_pipeline_manager()
    pipeline_manager.unload()
    
    return {"status": "success", "message": "Model unloaded"}