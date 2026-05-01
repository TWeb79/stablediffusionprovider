# =============================================================================
# Stable Diffusion API Service - Dockerfile (CPU-only, ARM compatible)
# Base: debian:12-slim (as per RULES_coding.md)
# Author: Inventions4All - github:TWeb79
# =============================================================================

FROM debian:12-slim

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*



# -----------------------------------------------------------------------------
# Python setup (venv instead of --break-system-packages)
# -----------------------------------------------------------------------------
RUN python3.11 -m venv /opt/venv

# Upgrade pip
RUN pip install --upgrade pip

# -----------------------------------------------------------------------------
# Install PyTorch (CPU, ARM-compatible)
# -----------------------------------------------------------------------------
RUN pip install \
    torch==2.2.2 \
    torchvision==0.17.2

# -----------------------------------------------------------------------------
# Install application dependencies
# -----------------------------------------------------------------------------
RUN pip install \
    "numpy<2" \          
    diffusers==0.25.0 \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    safetensors==0.4.1 \
    "huggingface_hub<0.20" \
    omegaconf==2.3.0 \   
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic==2.5.3 \
    pydantic-settings==2.1.0 \
    python-multipart==0.0.6 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.0 \
    Pillow>=10.0.0

# -----------------------------------------------------------------------------
# Security: non-root user
# -----------------------------------------------------------------------------
RUN groupadd -r sdprovider && useradd -r -g sdprovider sdprovider

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------
RUN mkdir -p /models /tmp/hf_cache && \
    chown -R sdprovider:sdprovider /models /tmp/hf_cache

# -----------------------------------------------------------------------------
# Copy application
# -----------------------------------------------------------------------------
COPY --chown=sdprovider:sdprovider src/ ./src/
COPY --chown=sdprovider:sdprovider config/ ./config/
COPY --chown=sdprovider:sdprovider requirements.txt ./

# -----------------------------------------------------------------------------
# Switch user
# -----------------------------------------------------------------------------
USER sdprovider

# -----------------------------------------------------------------------------
# HuggingFace cache
# -----------------------------------------------------------------------------
ENV HF_HOME=/tmp/hf_cache

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
EXPOSE 8141

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD \
    curl -f http://localhost:8141/health || exit 1

# -----------------------------------------------------------------------------
# Start
# -----------------------------------------------------------------------------
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8141"]