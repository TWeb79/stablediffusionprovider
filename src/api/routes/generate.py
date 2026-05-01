"""
Generate endpoint for the Stable Diffusion API.

Author: Inventions4All - github:TWeb79
"""

import io
import logging
import os
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.schemas.generate import GenerateRequest
from src.core.pipeline import get_pipeline_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Generation"])


@router.get(
    "/generate",
    summary="Generate Image",
    description="Generates an image from a text prompt.",
)
async def generate_image_get(
    prompt: str,
    negative_prompt: str = "",
    steps: int = 25,
    guidance: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = 0,
    model_path: str | None = None,
    explicit: bool = False,
) -> StreamingResponse:
    """
    Generate an image from a text prompt using query parameters.
    
    Args:
        prompt: Text description of the desired image
        negative_prompt: Things to avoid in the image
        steps: Number of inference steps (1-150)
        guidance: Guidance scale (1.0-20.0)
        width: Image width (256-1024, must be multiple of 8)
        height: Image height (256-1024, must be multiple of 8)
        seed: Random seed (0 = random)
        model_path: Full path to model file (optional)
        explicit: Use explicit content model instead of default
        
    Returns:
        PNG image stream
        
    Raises:
        HTTPException: If generation fails
    """
    return await _generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        seed=seed,
        model_path=model_path,
        explicit=explicit,
    )


@router.post(
    "/generate",
    summary="Generate Image (JSON)",
    description="Generates an image from a JSON request body.",
)
async def generate_image_json(request: GenerateRequest) -> StreamingResponse:
    """
    Generate an image from a JSON request body.
    
    Args:
        request: GenerateRequest with all generation parameters
        
    Returns:
        PNG image stream
    """
    return await _generate_image(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        steps=request.steps,
        guidance=request.guidance,
        width=request.width,
        height=request.height,
        seed=request.seed,
        model_path=request.model_path,
        explicit=request.explicit,
    )


async def _generate_image(
    prompt: str,
    negative_prompt: str = "",
    steps: int = 25,
    guidance: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = 0,
    model_path: str | None = None,
    explicit: bool = False,
) -> StreamingResponse:
    """
    Internal function to generate an image.
    
    Args:
        prompt: Text description of the desired image
        negative_prompt: Things to avoid in the image
        steps: Number of inference steps (1-150)
        guidance: Guidance scale (1.0-20.0)
        width: Image width (256-1024, must be multiple of 8)
        height: Image height (256-1024, must be multiple of 8)
        seed: Random seed (0 = random)
        model_path: Full path to model file (optional)
        explicit: Use explicit content model instead of default
        
    Returns:
        PNG image stream
        
    Raises:
        HTTPException: If generation fails
    """
    # Validate dimensions are multiples of 8 for Stable Diffusion compatibility
    if width % 8 != 0 or height % 8 != 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "Image dimensions must be multiples of 8"},
        )
    
    pipeline_manager = get_pipeline_manager()
    
    # Determine which model to use based on explicit flag
    final_model_path = model_path
    if not final_model_path and explicit:
        final_model_path = "/models/explicit.safetensors"
    elif not final_model_path:
        from ..core.config import get_settings
        settings = get_settings()
        final_model_path = settings.model.default_model
    
    # Validate model_path if provided
    if final_model_path and not os.path.isfile(final_model_path):
        logger.error(f"Model file not found at path: {final_model_path}")
        raise HTTPException(
            status_code=400,
            detail={"error": f"Model file not found at path: {final_model_path}"},
        )
    
    # Load model if not already loaded or if different model requested
    if not pipeline_manager.is_loaded or (final_model_path and final_model_path != pipeline_manager.current_model):
        try:
            pipeline_manager.load_model(final_model_path)
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise HTTPException(
                status_code=400,
                detail={"error": f"Model file not found at path: {final_model_path}"},
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(
                status_code=500,
                detail={"error": f"Failed to load model: {str(e)}"},
            )
    
    # Generate image with timing
    start_time = time.time()
    
    try:
        image = pipeline_manager.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Generation failed: {str(e)}"},
        )
    
    generation_time = (time.time() - start_time) * 1000
    
    logger.info(
        f"Generated image in {generation_time:.0f}ms "
        f"({width}x{height}, steps={steps})"
    )
    
    # Convert PIL Image to PNG bytes for streaming response
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    return StreamingResponse(
        img_bytes,
        media_type="image/png",
        headers={
            "X-Generation-Time-Ms": str(round(generation_time)),
            "X-Model": pipeline_manager.current_model or "unknown",
        },
    )