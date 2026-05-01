"""
Generate endpoint for the Stable Diffusion API.

Author: Inventions4All - github:TWeb79
"""

import io
import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ...schemas.generate import GenerateRequest, GenerateResponse
from ...core.pipeline import get_pipeline_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Generation"])


@router.get(
    "/generate",
    summary="Generate Image",
    description="Generates an image from a text prompt.",
)
async def generate_image(
    prompt: str,
    negative_prompt: str = "",
    steps: int = 25,
    guidance: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = 0,
    model: str | None = None,
) -> StreamingResponse:
    """
    Generate an image from a text prompt.
    
    Args:
        prompt: Text description of the desired image
        negative_prompt: Things to avoid in the image
        steps: Number of inference steps (1-150)
        guidance: Guidance scale (1.0-20.0)
        width: Image width (256-1024, must be multiple of 8)
        height: Image height (256-1024, must be multiple of 8)
        seed: Random seed (0 = random)
        model: Optional model name to use
        
    Returns:
        PNG image stream
        
    Raises:
        HTTPException: If generation fails
    """
    # Validate dimensions
    if width % 8 != 0 or height % 8 != 0:
        raise HTTPException(
            status_code=400,
            detail="Image dimensions must be multiples of 8",
        )
    
    pipeline_manager = get_pipeline_manager()
    
    # Load model if not already loaded
    if not pipeline_manager.is_loaded:
        try:
            pipeline_manager.load_model(model)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}",
            )
    elif model and model != pipeline_manager.current_model:
        # Switch model if different
        try:
            pipeline_manager.load_model(model)
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to switch model: {str(e)}",
            )
    
    # Generate image
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
            detail=f"Generation failed: {str(e)}",
        )
    
    generation_time = (time.time() - start_time) * 1000
    
    logger.info(
        f"Generated image in {generation_time:.0f}ms "
        f"({width}x{height}, steps={steps})"
    )
    
    # Convert to PNG bytes
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
    return await generate_image(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        steps=request.steps,
        guidance=request.guidance,
        width=request.width,
        height=request.height,
        seed=request.seed,
        model=request.model,
    )