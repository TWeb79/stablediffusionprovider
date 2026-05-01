# =============================================================================
# Stable Diffusion API Service - Dockerfile (CUDA-enabled)
# Author: Inventions4All - github:TWeb79
# =============================================================================

# Use PyTorch official image with CUDA 11.8 and cuDNN 8 runtime
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies
# Note: torch and torchvision are already included in base image
RUN pip install --no-cache-dir \
    diffusers==0.25.0 \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    safetensors==0.4.1 \
    "huggingface_hub<0.20" \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic==2.5.3 \
    pydantic-settings==2.1.0 \
    python-multipart==0.0.6 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.0 \
    Pillow>=10.0.0

# Create non-root user for security
RUN groupadd -r sdprovider && useradd -r -g sdprovider sdprovider

# Create model directory
RUN mkdir -p /models && chown -R sdprovider:sdprovider /models

# Copy application code
COPY --chown=sdprovider:sdprovider src/ ./src/
COPY --chown=sdprovider:sdprovider config/ ./config/
COPY --chown=sdprovider:sdprovider requirements.txt ./

# Switch to non-root user
USER sdprovider

# Set cache directory for HuggingFace
ENV HF_HOME=/tmp/hf_cache
RUN mkdir -p /tmp/hf_cache && chmod 777 /tmp/hf_cache

# Expose API port (8141 for project 41)
EXPOSE 8141

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    curl -f http://localhost:8141/health || exit 1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8141"]
