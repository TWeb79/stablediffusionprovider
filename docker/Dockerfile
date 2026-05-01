# =============================================================================
# Stable Diffusion API Service - Dockerfile
# Author: Inventions4All - github:TWeb79
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile native libraries
# -----------------------------------------------------------------------------
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies for PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install --no-cache-dir \
    diffusers==0.27.0 \
    transformers==4.40.0 \
    accelerate==0.30.0 \
    safetensors==0.4.3

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as runtime

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install FastAPI and web server
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.0 \
    pydantic==2.7.0 \
    pydantic-settings==2.2.1 \
    python-multipart==0.0.9 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.1

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

# Expose API port (8141 for project 41)
EXPOSE 8141

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8141/health || exit 1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8141"]