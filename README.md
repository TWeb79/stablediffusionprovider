# Stable Diffusion API Provider

A Docker-based REST API service for Stable Diffusion image generation with GPU acceleration support.

**Author:** Inventions4All - github:TWeb79

---

## Overview

This service provides a FastAPI-based REST API for generating images using Stable Diffusion models. It supports:

- GPU acceleration (NVIDIA CUDA) with automatic CPU fallback
- Model loading from a central folder (no copying required)
- Memory optimization (attention slicing, CPU offload)
- REST API for image generation
- Auto-generated API documentation

---

## Service Ports

| Port | Service Type | Description |
|------|-------------|-------------|
| 8141 | FastAPI | Main API service |

---

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support (for GPU acceleration)
- Stable Diffusion model files (.safetensors or .ckpt)
- Model directory mounted at `./dev/ai/external/_Models/Stable-diffusion`

### Running with Docker Compose

```bash
# Build and start the service
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop the service
docker-compose -f docker/docker-compose.yml down
```

### Running Locally (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MODEL_DIR=./models
export DEVICE=cuda  # or cpu
export API_PORT=8141

# Run the service
python -m src.main
```

---

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service health status and system information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "loaded_model": "model.safetensors",
  "device": "NVIDIA RTX 3090",
  "device_type": "cuda",
  "cuda_available": true,
  "cuda_device_count": 1
}
```

### List Models

```bash
GET /models
```

Returns list of available models in the model directory.

**Response:**
```json
{
  "models": [
    {
      "name": "model.safetensors",
      "path": "/models/model.safetensors",
      "size_bytes": 4265382912,
      "size_mb": 4067.5,
      "format": "safetensors"
    }
  ],
  "count": 1,
  "current_model": "model.safetensors"
}
```

### Load Model

```bash
POST /models/{model_name}/load
```

Loads a specific model into memory.

### Generate Image

```bash
GET /generate?prompt=a beautiful landscape
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | string | required | Text description of the desired image |
| negative_prompt | string | "" | Things to avoid in the image |
| steps | int | 25 | Number of inference steps (1-150) |
| guidance | float | 7.5 | Guidance scale (1.0-20.0) |
| width | int | 512 | Image width (256-1024, multiple of 8) |
| height | int | 512 | Image height (256-1024, multiple of 8) |
| seed | int | 0 | Random seed (0 = random) |
| model | string | null | Optional model name to use |

**Response:** PNG image binary

### Example Request

```bash
# Generate an image using curl
curl -o output.png "http://localhost:8141/generate?prompt=a%20beautiful%20sunset%20over%20mountains&steps=30&width=768&height=512"

# With negative prompt
curl -o output.png "http://localhost:8141/generate?prompt=a%20cat&negative_prompt=blurry&steps=25"
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 0.0.0.0 | API server host |
| `API_PORT` | 8141 | API server port |
| `MODEL_DIR` | /models | Path to model files |
| `DEFAULT_MODEL` | auto | Default model to load |
| `DEVICE` | cuda | Compute device (cuda/cpu) |
| `DEFAULT_STEPS` | 25 | Default inference steps |
| `DEFAULT_GUIDANCE` | 7.5 | Default guidance scale |
| `DEFAULT_WIDTH` | 512 | Default image width |
| `DEFAULT_HEIGHT` | 512 | Default image height |
| `HF_TOKEN` | - | HuggingFace token (optional) |
| `LOG_LEVEL` | INFO | Logging level |

### GPU Memory Requirements

| Resolution | Recommended VRAM |
|------------|-----------------|
| 512x512 | 6-8 GB |
| 768x768 | 8-10 GB |
| 1024x1024 | 12+ GB |

---

## API Documentation

Interactive API documentation is available at:

- Swagger UI: http://localhost:8141/docs
- ReDoc: http://localhost:8141/redoc
- OpenAPI JSON: http://localhost:8141/openapi.json

---

## Model Formats

Supported model formats:

- `.safetensors` (recommended)
- `.ckpt`

Place model files in the mounted model directory. The service will automatically discover and list available models.

---

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

1. Enable attention slicing: `ATTENTION_SLICING=true`
2. Enable CPU offload: `CPU_OFFLOAD=true`
3. Reduce image resolution
4. Reduce batch size

### Model Not Found

Ensure your model directory is correctly mounted and contains `.safetensors` or `.ckpt` files.

### Slow Generation on CPU

CPU generation is significantly slower than GPU. For best performance, use a GPU with at least 6GB VRAM.

---

## License

MIT License

---

## Author

Inventions4All - github:TWeb79