# Stable Diffusion API Provider

A Docker-based REST API service for Stable Diffusion image generation optimized for **CPU-only** execution in Debian 12 slim containers on Apple Silicon hosts. A dedicated `run_local.py` entry point enables Apple Metal (MPS) acceleration when running directly on macOS.

**Author:** Inventions4All - github:TWeb79

---

## Overview

This service provides a FastAPI-based REST API for generating images using Stable Diffusion models. It supports:

- CPU-only inference in Docker (Debian 12 slim, ARM64)
- Apple Metal (MPS) acceleration via `run_local.py` when running directly on macOS
- Model loading from a central folder (no copying required)
- Memory optimization (attention slicing, VAE slicing, CPU offload)
- Auto-generated API documentation

---

## Service Ports

| Port | Service Type | Description |
|------|-------------|-------------|
| 8141 | FastAPI | Main API service |

---

## Quick Start

### Prerequisites

- Docker Desktop on macOS (Apple Silicon)
- Stable Diffusion model files (.safetensors or .ckpt)
- Model directory mounted at `./docker/dev/ai/external/_Models/Stable-diffusion`

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

# Run with CPU (default)
python -m src.main

# Run with Apple Metal (MPS) using helper
python run_local.py --model-dir ./docker/dev/ai/external/_Models/Stable-diffusion --port 8141
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
  "timestamp": "2026-05-01T00:00:00Z",
  "loaded_model": "model.safetensors",
  "device": "CPU",
  "device_type": "cpu",
  "mps_available": false,
  "torch_num_threads": 8,
  "torch_interop_threads": 4
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
| `DEVICE` | cpu | Compute device (cpu/mps/auto). `auto` prefers MPS when available locally. |
| `TORCH_NUM_THREADS` | (unset) | Optional override for `torch.set_num_threads` |
| `TORCH_INTEROP_THREADS` | (unset) | Optional override for `torch.set_num_interop_threads` |
| `OMP_NUM_THREADS` | (unset) | Optional CPU threading hint |
| `MKL_NUM_THREADS` | (unset) | Optional MKL threading hint |
| `DEFAULT_STEPS` | 25 | Default inference steps |
| `DEFAULT_GUIDANCE` | 7.5 | Default guidance scale |
| `DEFAULT_WIDTH` | 512 | Default image width |
| `DEFAULT_HEIGHT` | 512 | Default image height |
| `HF_TOKEN` | - | HuggingFace token (optional) |
| `LOG_LEVEL` | INFO | Logging level |

### CPU Performance Tips

- Keep image sizes at 512x512 or 640x640 for fastest CPU inference
- Set `TORCH_NUM_THREADS` to the number of physical cores (e.g., 8 on M2 Pro)
- Enable sequential CPU offload for lower memory usage (default in Docker)

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

### Slow Generation on CPU

1. Reduce `steps` (e.g., 20)
2. Lower resolution to 448x448 or 512x512
3. Increase `TORCH_NUM_THREADS`/`OMP_NUM_THREADS`
4. Batch size must remain 1 for CPU workloads

### Model Not Found

Ensure your model directory is correctly mounted and contains `.safetensors` or `.ckpt` files.

### Apple Metal (MPS) Issues

- Ensure you run `python run_local.py` outside Docker
- Requires PyTorch with MPS support (automatically installed via `requirements.txt`)

---

## License

MIT License

---

## Author

Inventions4All - github:TWeb79

---

### Docker Deploy
docker-compose -f docker/docker-compose.yml build --no-cache
docker-compose -f docker/docker-compose.yml up -d