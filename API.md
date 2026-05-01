# Stable Diffusion API Documentation

A FastAPI-based REST API service for Stable Diffusion image generation optimized for CPU-only execution in Docker containers with optional Apple Metal (MPS) acceleration for local macOS runs.
https://github.com/TWeb79/stablediffusionprovider

**Author:** Inventions4All - github:TWeb79

## Table of Contents

- [Overview](#overview)
- [How Model Switching Works](#how-model-switching-works)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
  - [Health Check](#health-check)
  - [List Models](#list-models)
  - [Load Model](#load-model)
  - [Load Model from Path](#load-model-from-path)
  - [Unload Model](#unload-model)
  - [Generate Image](#generate-image)
- [Request/Response Examples](#requestresponse-examples)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

---

## Overview

The Stable Diffusion API Provider provides a RESTful interface for generating images using Stable Diffusion models. It supports:

- **CPU-only inference** in Docker (Debian 12 slim, ARM64)
- **Apple Metal (MPS) acceleration** when running locally on macOS
- **Model Management**: Load, unload, and switch between multiple models
- **Memory Optimization**: Attention slicing, VAE slicing, and CPU offload options
- **Flexible Generation**: Configurable steps, guidance scale, dimensions, and seeds

### How Model Switching Works

Models are loaded on-demand and cached in memory. When you request a generation with a specific `model_path`:

1. If the model is **already loaded** (same path), it's reused immediately.
2. If the model is **cached** (previously loaded but currently unloaded), it's reloaded from cache.
3. If the model is **not cached**, it's loaded from the file system and cached for future use.

The cache is stored in memory until the service restarts. You can explicitly unload models to free memory using the `/models/unload` endpoint.

The default model (if `DEFAULT_MODEL` environment variable is set) is automatically loaded on the first generation request if no `model_path` is provided.

---

## Quick Start

### Using Docker Compose

```bash
# Start the service (CPU-only mode)
docker-compose -f docker/docker-compose.yml up -d

# Check health
curl http://localhost:8141/health

# Generate an image with default model
curl "http://localhost:8141/generate?prompt=a%20beautiful%20landscape"

# Load a specific model
curl -X POST "http://localhost:8141/load-model" \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/models/v1-5-pruned.safetensors"}'

# Generate using that model
curl "http://localhost:8141/generate?prompt=a%20futuristic%20city&model_path=/models/v1-5-pruned.safetensors" \
  -o city.png
```

### Using run_local.py (macOS with MPS)

```bash
# Install dependencies
pip install -r requirements.txt

# Run with MPS acceleration (auto-detected)
python run_local.py --model-dir ./models --port 8141

# Or force CPU mode
python run_local.py --device cpu --model-dir ./models
```

### Using Python Directly

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (CPU mode)
DEVICE=cpu python -m src.main

# Or with MPS (on macOS)
DEVICE=auto python -m src.main
```

---

## API Endpoints

### Health Check

**GET** `/health`

Returns service health status and system information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "loaded_model": "v1-5-pruned.safetensors",
  "device": "CPU",
  "device_type": "cpu",
  "mps_available": true,
  "torch_num_threads": 8,
  "torch_interop_threads": 4
}
```

---

### List Models

**GET** `/models`

Returns a list of all available Stable Diffusion models in the model directory.

**Response:**
```json
{
  "models": [
    {
      "name": "v1-5-pruned.safetensors",
      "path": "/models/v1-5-pruned.safetensors",
      "size_bytes": 4265382912,
      "size_mb": 4067.52,
      "format": "safetensors"
    }
  ],
  "count": 1,
  "current_model": "v1-5-pruned.safetensors"
}
```

---

### Load Model

**POST** `/models/{model_name}/load`

Loads a specific model into memory by name.

**Path Parameters:**
- `model_name` (string, required): Name of the model file to load

**Response:**
```json
{
  "success": true,
  "model": "v1-5-pruned.safetensors",
  "device": "CPU",
  "memory_optimizations": {
    "attention_slicing": true,
    "cpu_offload": true
  }
}
```

**Error Response (404):**
```json
{
  "detail": "Model not found: nonexistent-model.safetensors"
}
```

---

### Load Model from Path

**POST** `/load-model`

Loads a specific model into memory using a full file path. This allows loading models from any location accessible inside the container.

**Request Body (JSON):**
```json
{
  "model_path": "/models/v1-5-pruned.safetensors"
}
```

**Response:**
```json
{
  "success": true,
  "model_path": "/models/v1-5-pruned.safetensors",
  "model": "v1-5-pruned.safetensors",
  "device": "CPU"
}
```

**Error Response (400):**
```json
{
  "detail": {
    "error": "Model file not found at path: /models/nonexistent.safetensors"
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8141/load-model" \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/models/v1-5-pruned.safetensors"}'
```

---

### Unload Model

**POST** `/models/unload`

Unloads the currently loaded model and frees memory.

**Response:**
```json
{
  "status": "success",
  "message": "Model unloaded"
}
```

---

### Generate Image

The generate endpoint supports two methods: GET (query parameters) and POST (JSON body).

#### GET Method

**GET** `/generate`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the desired image |
| `negative_prompt` | string | "" | Things to avoid in the image |
| `steps` | integer | 25 | Number of inference steps (1-150) |
| `guidance` | float | 7.5 | Guidance scale (1.0-20.0) |
| `width` | integer | 512 | Image width (256-1024, must be multiple of 8) |
| `height` | integer | 512 | Image height (256-1024, must be multiple of 8) |
| `seed` | integer | 0 | Random seed (0 = random) |
| `model_path` | string | null | Full path to model file (e.g., /models/model.safetensors) |

**Response:** PNG image binary

**Response Headers:**
- `X-Generation-Time-Ms`: Time taken for generation in milliseconds
- `X-Model`: Name of the model used

**Example:**
```bash
curl "http://localhost:8141/generate?prompt=a%20cat%20wearing%20a%20hat&steps=20&width=512&height=512" \
  -o output.png
```

**Example with model_path:**
```bash
curl "http://localhost:8141/generate?prompt=a%20futuristic%20city&model_path=/models/v1-5-pruned.safetensors" \
  -o city.png
```

#### POST Method

**POST** `/generate`

**Request Body (JSON):**
```json
{
  "prompt": "a beautiful sunset over mountains",
  "negative_prompt": "blurry, low quality, distorted",
  "steps": 30,
  "guidance": 7.5,
  "width": 768,
  "height": 512,
  "seed": 42,
  "model_path": "/models/v1-5-pruned.safetensors"
}
```

**Response:** PNG image binary

**Example:**
```bash
curl -X POST "http://localhost:8141/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a futuristic city at sunset",
    "model_path": "/models/v1-5-pruned.safetensors",
    "steps": 30,
    "guidance": 7.5
  }' \
  -o city.png
```

---

## Request/Response Examples

### Generate with All Parameters (GET)

```bash
curl "http://localhost:8141/generate?\
prompt=a%20cyberpunk%20city%20at%20night%20with%20neon%20lights&\
negative_prompt=daytime,%20blurry,%20low%20quality&\
steps=30&\
guidance=8.0&\
width=768&\
height=512&\
seed=12345" \
  -o cyberpunk_city.png
```

### Generate with JSON (POST)

```bash
curl -X POST "http://localhost:8141/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a serene lake surrounded by mountains",
    "negative_prompt": "people, buildings, roads",
    "steps": 25,
    "guidance": 7.5,
    "width": 512,
    "height": 512,
    "seed": 999
  }' \
  -o lake_mountains.png
```

### Python Client Example

```python
import requests
import json

# Generate image using GET
response = requests.get(
    "http://localhost:8141/generate",
    params={
        "prompt": "a beautiful landscape",
        "steps": 25,
        "guidance": 7.5,
    }
)

with open("output.png", "wb") as f:
    f.write(response.content)

# Generate image using POST
data = {
    "prompt": "a futuristic city",
    "negative_prompt": "old, blurry",
    "steps": 30,
    "guidance": 8.0,
    "width": 768,
    "height": 512,
    "seed": 42
}

response = requests.post(
    "http://localhost:8141/generate",
    json=data
)

with open("future_city.png", "wb") as f:
    f.write(response.content)

# Check generation time from headers
print(f"Generation time: {response.headers.get('X-Generation-Time-Ms')}ms")
print(f"Model used: {response.headers.get('X-Model')}")
```

### JavaScript/Node.js Example

```javascript
// Using fetch API
const response = await fetch('http://localhost:8141/generate', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        prompt: 'a beautiful sunset',
        steps: 25,
        guidance: 7.5,
        width: 512,
        height: 512,
    }),
});

const blob = await response.blob();
const url = URL.createObjectURL(blob);

// Create download link
const a = document.createElement('a');
a.href = url;
a.download = 'generated.png';
a.click();
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 0.0.0.0 | Host to bind the API server |
| `API_PORT` | 8141 | Port to bind the API server |
| `API_WORKERS` | 1 | Number of worker processes |
| `MODEL_DIR` | /models | Directory containing model files |
| `DEFAULT_MODEL` | (none) | Default model to load (e.g., `/models/default.safetensors`). If not set and no `model_path` is provided per request, the service will auto-detect the first model in `MODEL_DIR`. |
| `SAFETY_CHECKER` | false | Enable safety checker |
| `DEVICE` | cpu | Compute device (cpu/mps/auto). `auto` prefers MPS when available locally. |
| `ATTENTION_SLICING` | true | Enable attention slicing |
| `CPU_OFFLOAD` | true | Enable sequential CPU offload |
| `TORCH_NUM_THREADS` | (auto) | Number of torch threads |
| `TORCH_INTEROP_THREADS` | (auto) | Number of torch interop threads |
| `OMP_NUM_THREADS` | (auto) | OMP thread count |
| `MKL_NUM_THREADS` | (auto) | MKL thread count |
| `DEFAULT_STEPS` | 25 | Default inference steps |
| `DEFAULT_GUIDANCE` | 7.5 | Default guidance scale |
| `DEFAULT_WIDTH` | 512 | Default image width |
| `DEFAULT_HEIGHT` | 512 | Default image height |
| `HF_TOKEN` | (none) | HuggingFace token for gated models |
| `LOG_LEVEL` | INFO | Logging level |

### Configuration File

Configuration can also be set via `config/config.yml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8141
  workers: 1

model:
  directory: "/models"
  default_model: "/models/v1-5-pruned.safetensors"
  safety_checker: false

device:
  device: "auto"  # auto, cpu, mps
  attention_slicing: true
  cpu_offload: true
  num_threads: 8
  interop_threads: 4

generation:
  default_steps: 25
  default_guidance: 7.5
  default_width: 512
  default_height: 512

logging:
  level: "INFO"
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters or missing file |
| 404 | Not Found - Model not found |
| 500 | Internal Server Error - Generation or loading failed |

### Error Response Format

All errors return a structured JSON object:

```json
{
  "error": "Model file not found at path: /models/missing.safetensors"
}
```

Some errors may include additional details:

```json
{
  "error": "Failed to load model",
  "detail": "Out of memory"
}
```

### Common Errors

**Invalid Dimensions:**
```json
{"error": "Image dimensions must be multiples of 8"}
```

**Model File Not Found:**
```json
{"error": "Model file not found at path: /models/nonexistent.safetensors"}
```

**Model Not Found (by name):**
```json
{"error": "Model not found: nonexistent-model.safetensors"}
```

**Generation Failed:**
```json
{"error": "Generation failed: Out of memory"}
```

**No Model Loaded:**
```json
{"error": "No model loaded. Call load_model() first."}
```

---

## Device Support

### Docker (CPU Only)

For Docker deployments on Apple Silicon or other ARM platforms:

```bash
# Using Docker Compose (recommended)
docker-compose -f docker/docker-compose.yml up -d

# Or manually
docker run -p 8141:8141 \
  -v /path/to/models:/models:ro \
  41-sdprovider-api:latest
```

### Apple Silicon (MPS)

On Apple Silicon Macs running locally (not in Docker), use `run_local.py` for MPS acceleration:

```bash
# Auto-detect MPS (preferred)
python run_local.py --model-dir ./models

# Force MPS
python run_local.py --device mps --model-dir ./models

# Force CPU
python run_local.py --device cpu --model-dir ./models
```

### CPU Only

For systems without GPU acceleration:

```bash
# Force CPU mode
DEVICE=cpu python -m src.main

# Or with custom thread settings
TORCH_NUM_THREADS=8 python -m src.main
```

---

## Performance Tips

### CPU Inference

- Keep image sizes at 512x512 or 640x640 for fastest inference
- Reduce `steps` to 20-25 for quick previews
- Increase `TORCH_NUM_THREADS` to match physical CPU cores
- CPU offload is enabled by default to reduce memory usage

### Apple Metal (MPS)

- Use float32 (default) for stability
- 768x768 images work well on M1/M2 chips
- Consider reducing steps if generation is slow

---

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8141/docs
- **ReDoc**: http://localhost:8141/redoc
- **OpenAPI JSON**: http://localhost:8141/openapi.json

---

## Rate Limits

Currently, there are no rate limits implemented. For production deployments, consider adding rate limiting middleware.

---

## License

MIT License