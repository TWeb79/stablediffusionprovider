# Implementation Plan: Local Stable Diffusion API Service

## Overview
Create a Docker-based Stable Diffusion service with a FastAPI backend that:
- Loads models from a central folder (no copying)
- Provides REST API for image generation
- Supports GPU acceleration (NVIDIA CUDA) with CPU fallback

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Docker Container                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   FastAPI   │───▶│  SD Pipeline│───▶│  Image Output   │  │
│  │   Service   │    │   Loader    │    │  (PNG Response) │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│         │                  │                                 │
│         ▼                  ▼                                 │
│  ┌─────────────┐    ┌─────────────┐                         │
│  │   Config    │    │   Model     │◀─── Volume Mount ───┐   │
│  │  (env/yml)  │    │   Cache     │                    │   │
│  └─────────────┘    └─────────────┘    ┌────────────────┴───┤
└─────────────────────────────────────────│ Host Model Folder │
                                          │ ./dev/ai/external │
                                          │ /_Models/Stable-  │
                                          │ diffusion/        │
                                          └───────────────────┘
```

---

## Project Structure

```
sdprovider/
├── docker/
│   ├── Dockerfile              # Main container image
│   └── docker-compose.yml       # Service orchestration
├── src/
│   ├── main.py                 # FastAPI application entry
│   ├── api/
│   │   ├── routes/
│   │   │   ├── generate.py     # /generate endpoint
│   │   │   ├── models.py        # /models endpoint
│   │   │   └── health.py        # /health endpoint
│   │   └── dependencies.py      # Shared dependencies
│   ├── core/
│   │   ├── config.py            # Configuration management
│   │   ├── pipeline.py          # SD Pipeline manager
│   │   └── device.py            # Device detection (CUDA/CPU)
│   └── schemas/
│       ├── generate.py          # Request/Response schemas
│       └── model.py             # Model info schemas
├── config/
│   └── config.yml              # Default configuration
├── tests/
│   ├── test_config.py
│   ├── test_pipeline.py
│   └── test_api.py
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── README.md                   # Setup instructions
└── ARCHITECTURE.md             # Architecture documentation
```

---

## Port Configuration

Based on RULES_ports.md for project 41:
- **8141** → FastAPI service (main service port)
- **8041** → Web dashboard (reserved for future)

---

## Implementation Steps

### Phase 1: Docker Configuration
1. **Create Dockerfile**
   - Base image: `nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04` (for GPU) or `python:3.10-slim` (CPU-only)
   - Multi-stage build for smaller image
   - Install PyTorch with CUDA support
   - Install diffusers, fastapi, uvicorn, pydantic
   - Expose port 8141

2. **Create docker-compose.yml**
   - Service: `sdprovider`
   - Volume mount: `./dev/ai/external/_Models/Stable-diffusion/:/models:ro`
   - GPU access: `deploy.resources.reservations.devices: gpu`
   - Environment variables for configuration
   - Health check configuration

### Phase 2: Core Application
3. **Configuration Management** (`src/core/config.py`)
   - Load from environment variables and config.yml
   - Model directory path (default: `/models`)
   - API settings (host, port 8141, workers)
   - Generation defaults (steps, guidance, dimensions)

4. **Device Detection** (`src/core/device.py`)
   - Detect CUDA availability
   - Fallback to CPU if no GPU
   - Memory optimization settings

5. **Pipeline Manager** (`src/core/pipeline.py`)
   - Lazy loading of models
   - Model switching capability
   - Memory management (attention slicing, CPU offload)
   - Scheduler configuration (DPM++ solver)

### Phase 3: API Endpoints
6. **Health Endpoint** (`/health`)
   - Service status
   - Loaded model info
   - Device info

7. **Models Endpoint** (`/models`)
   - List available models from mounted folder
   - Model metadata (filename, size)
   - Currently loaded model

8. **Generate Endpoint** (`/generate`)
   - Parameters: prompt, negative_prompt, steps, guidance, width, height, seed, model (optional)
   - Returns: PNG image stream
   - Request validation with Pydantic

### Phase 4: Model Management
9. **Model Discovery**
   - Scan `/models` directory for `.safetensors` and `.ckpt` files
   - Cache model list
   - Support model selection by filename

10. **Model Loading**
    - Use `from_single_file()` for .safetensors
    - Use `from_pretrained()` with safety checker disabled
    - Configure scheduler (DPM++ for quality)

### Phase 5: Testing & Documentation
11. **Testing**
    - Unit tests for config, pipeline, API
    - Integration test with sample generation

12. **Documentation**
    - README with setup instructions
    - API documentation (auto-generated via FastAPI/OpenAPI)
    - Environment variable reference

---

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/models` | Path to model files (volume mount) |
| `DEFAULT_MODEL` | auto-detect | Default model to load |
| `DEVICE` | `cuda`/`cpu` | Compute device |
| `DEFAULT_STEPS` | `25` | Default inference steps |
| `DEFAULT_GUIDANCE` | `7.5` | Default guidance scale |
| `DEFAULT_WIDTH` | `512` | Default image width |
| `DEFAULT_HEIGHT` | `512` | Default image height |
| `HF_TOKEN` | - | HuggingFace token (optional, for gated models) |
| `API_PORT` | `8141` | FastAPI service port |

---

## API Reference

### `GET /health`
Returns service health and status.

### `GET /models`
Returns list of available models.

### `POST /models/{model_name}/load`
Loads a specific model into memory.

### `GET /generate`
Generates an image from a text prompt.

**Query Parameters:**
- `prompt` (required): Text description
- `negative_prompt` (optional): Things to avoid
- `steps` (optional, default=25): Inference steps
- `guidance` (optional, default=7.5): Guidance scale
- `width` (optional, default=512): Image width
- `height` (optional, default=512): Image height
- `seed` (optional, default=0): Random seed (0 = random)

**Response:** PNG image binary

---

## Dependencies

```
torch>=2.0.0
diffusers>=0.25.0
transformers>=4.30.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
python-multipart>=0.0.6
pyyaml>=6.0
python-dotenv>=1.0.0
```

---

## Notes

1. **GPU Memory**: For 512x512 images with default settings, ~6-8GB VRAM recommended
2. **Model Format**: Primary support for `.safetensors` (recommended) and `.ckpt`
3. **Security**: Safety checker disabled by default (configurable)
4. **Performance**: Sequential CPU offload enabled for better RAM usage
5. **Author**: Inventions4All - github:TWeb79