# Implementation Plan: Local Stable Diffusion API Service

## Overview
Create a Docker-based Stable Diffusion service with a FastAPI backend that:
- Loads models from a central folder (no copying)
- Provides REST API for image generation
- Runs entirely on CPU when packaged in Docker for Debian 12 slim on ARM
- Offers a dedicated local runner (run_local.py) that leverages Apple Metal (MPS) when available

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

### Phase 1: Implementation Plan & Documentation Updates
1. Align implementationplan.md, FIX.md, README.md, API.md, and ARCHITECTURE.md with CPU-only Docker deployment plus local Apple MPS runner requirements.
2. Ensure author attribution remains "Inventions4All - github:TWeb79" across docs.
3. Document run_local.py usage and the distinction between Docker-on-ARM CPU mode vs. local MPS support.

### Phase 2: Docker & Runtime Configuration
4. **Dockerfile**
   - Base image: `debian:12-slim` (mandatory per RULES_coding)
   - Install Python 3.10+, PyTorch CPU wheels, diffusers, FastAPI stack
   - Configure optimal CPU runtime flags (OMP_NUM_THREADS etc.)
   - Expose port 8141

5. **docker-compose.yml**
   - Service name: `41-sdprovider-api`
   - Bind mount `./docker/dev/ai/external/_Models/Stable-diffusion/:/models:ro`
   - CPU-only execution (no GPU devices)
   - Healthcheck hitting `/health`

### Phase 3: Core Application Refactor
6. **Configuration Management** (`src/core/config.py`)
   - Default device = `cpu`
   - Support `DEVICE=cpu|mps|auto` but never `cuda`
   - Provide CPU thread configuration options

7. **Device Handling** (`src/core/device.py`)
   - Remove CUDA detection logic
   - Support CPU + optional MPS detection for local run
   - Return deterministic CPU-first DeviceInfo structure

8. **Pipeline Manager** (`src/core/pipeline.py`)
   - Force `torch_dtype=torch.float32`
   - Apply CPU-friendly optimizations (attention/vae slicing, sequential CPU offload)
   - Wrap generation in `torch.inference_mode()`
   - Prevent `.to("cuda")` usage; use `.to("cpu")` or keep on CPU

9. **run_local.py**
   - Entry point for local Apple Silicon execution
   - Detect MPS via `torch.backends.mps.is_available()`
   - Configure threads, load models once, expose CLI to trigger generation or start API

### Phase 4: API & Schema Consistency
10. Update `/health`, `/models`, `/load-model`, `/generate` routes and schemas to reference CPU/MPS devices only.
11. Ensure routes remain thin; move any remaining business logic to services/core modules if needed.

### Phase 5: Testing & Validation
12. Update tests to reflect CPU defaults (e.g., `tests/test_device.py`, `tests/test_api.py`).
13. Add tests for run_local runner helpers if applicable.
14. Execute `pytest` (or document reasoning if runtime not available) to confirm CPU-first logic.

### Phase 6: Documentation & Release Prep
15. Document CPU tuning tips (thread counts, env vars) in README/API docs.
16. Add instructions for Docker ARM deployment vs. local MPS run.
17. Confirm `.gitignore` remains up to date.

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