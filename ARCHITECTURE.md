# Stable Diffusion API Provider - Architecture

**Author:** Inventions4All - github:TWeb79

---

## System Overview

The SDProvider is a Docker-based REST API service for Stable Diffusion image generation. It provides a clean separation between the API layer, core business logic, and model management.

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

## Module Responsibilities

### `src/main.py`
- FastAPI application entry point
- Lifespan management (startup/shutdown)
- Router registration
- CORS middleware configuration

### `src/core/`
Core business logic layer - no API dependencies.

#### `config.py`
- Configuration management using Pydantic Settings
- Loads from environment variables and YAML config
- Provides type-safe settings with validation
- Singleton pattern for global settings access

#### `device.py`
- CUDA/CPU device detection
- Memory requirement estimation
- Device optimization settings

#### `pipeline.py`
- Stable Diffusion pipeline lifecycle management
- Model loading from .safetensors and .ckpt files
- Memory optimization (attention slicing, CPU offload)
- Image generation with DPM++ scheduler

### `src/api/`
API layer - depends on core layer.

#### `routes/`
- `health.py` - Service health check endpoint
- `models.py` - Model discovery and loading endpoints
- `generate.py` - Image generation endpoint

#### `dependencies.py`
- Shared FastAPI dependencies
- Dependency injection helpers

### `src/schemas/`
Pydantic models for request/response validation.

- `health.py` - Health check response schema
- `model.py` - Model info and list schemas
- `generate.py` - Generation request/response schemas

---

## Data Flow

### Image Generation Flow

```
1. Client sends request to /generate
   │
2. FastAPI validates request parameters
   │
3. PipelineManager checks if model is loaded
   │   └── If not loaded, auto-detect and load first model
   │
4. Generate image using SD pipeline
   │   - Set random seed for reproducibility
   │   - Run inference with configured steps
   │
5. Convert PIL Image to PNG bytes
   │
6. Return StreamingResponse with PNG data
```

### Model Loading Flow

```
1. Client sends POST /models/{name}/load
   │
2. Check if model already loaded (return early)
   │
3. Check model cache (return if cached)
   │
4. Find model file in model directory
   │
5. Load pipeline based on file format
   │   ├── .safetensors → from_pretrained + load weights
   │   └── .ckpt → from_single_file
   │
6. Apply memory optimizations
   │   ├── Attention slicing
   │   ├── VAE slicing
   │   └── CPU offload (if enabled)
   │
7. Configure DPM++ scheduler
   │
8. Cache pipeline and return
```

---

## External Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| torch | PyTorch ML framework | 2.2.0 |
| diffusers | Stable Diffusion pipelines | 0.27.0 |
| transformers | Model utilities | 4.40.0 |
| fastapi | Web framework | 0.111.0 |
| uvicorn | ASGI server | 0.30.0 |
| pydantic | Data validation | 2.7.0 |

---

## Service Boundaries

### API Layer (src/api/)
- Handles HTTP requests/responses
- Input validation
- Error handling and HTTP status codes
- Does NOT contain business logic

### Core Layer (src/core/)
- Business logic implementation
- Model management
- Configuration handling
- Device management
- Does NOT depend on API layer

### Schema Layer (src/schemas/)
- Data transfer objects
- Request/response models
- Pure Pydantic models

---

## Configuration Hierarchy

Configuration values are loaded in this priority order (highest to lowest):

1. Environment variables
2. .env file
3. config/config.yml
4. Default values in Pydantic models

---

## Memory Optimization Strategy

The service uses multiple strategies to reduce VRAM usage:

1. **Attention Slicing**: Splits attention computation into smaller parts
2. **VAE Slicing**: Processes VAE in slices for large images
3. **Sequential CPU Offload**: Moves model components to CPU when not in use

These can be configured via environment variables:
- `ATTENTION_SLICING=true` (default)
- `CPU_OFFLOAD=false` (default)

---

## Port Configuration

Following RULES_ports.md for project 41:

| Port | Service | Description |
|------|---------|-------------|
| 8141 | FastAPI | Main API service |

---

## Security Considerations

1. **Read-only model mount**: Models are mounted as read-only (:ro)
2. **Non-root container user**: Runs as unprivileged user
3. **Safety checker disabled by default**: Configurable via `SAFETY_CHECKER=true`
4. **CORS configured for development**: Restrict origins in production

---

## Future Extensions

Potential areas for expansion:

1. **Batch generation**: Generate multiple images in one request
2. **Model switching**: Hot-swap models without restart
3. **Caching**: Cache generated images
4. **WebSocket support**: Real-time generation progress
5. **Authentication**: API key or OAuth for access control