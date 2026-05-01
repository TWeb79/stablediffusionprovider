"""
Stable Diffusion API Service - Main Application Entry Point.

A FastAPI-based REST API for Stable Diffusion image generation.
Supports GPU acceleration (NVIDIA CUDA) with CPU fallback.

Author: Inventions4All - github:TWeb79
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import generate, health, load, models
from src.core.config import get_settings

import warnings
import logging

# Suppress noisy deprecation warnings from older diffusers/transformers builds
warnings.filterwarnings(
    "ignore",
    message="torch.utils._pytree._register_pytree_node is deprecated",
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    logger.info(f"Starting SDProvider API on port {settings.api.port}")
    logger.info(f"Model directory: {settings.model.directory}")
    logger.info(f"Device: {settings.device.device}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SDProvider API")


# Create FastAPI application
app = FastAPI(
    title="Stable Diffusion API Provider",
    description=(
        "A REST API service for Stable Diffusion image generation. "
        "Supports GPU acceleration with automatic CPU fallback."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware for web dashboard compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(load.router)
app.include_router(models.router)
app.include_router(generate.router)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with service information.
    
    Returns:
        Service info and links to documentation
    """
    return {
        "service": "Stable Diffusion API Provider",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
        "generate": "/generate",
    }


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )