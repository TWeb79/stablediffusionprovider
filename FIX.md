You are a senior Python & ML engineer responsible for making the Stable Diffusion provider production-ready on CPU in Docker and performant on Apple Silicon locally.

## Primary Goal (Docker on ARM CPU)

Refactor the entire codebase so it runs on **CPU only** inside a Docker container based on **Debian 12 slim** targeting **ARM (Apple Silicon)**. CUDA is unavailable and must not be referenced.

### Mandatory Actions
1. **Eliminate GPU logic**
   - Remove `.cuda()`, `.to("cuda")`, CUDA environment variables, and any GPU health reporting.
   - Delete `torch.cuda.*` conditionals; device detection must default to CPU.

2. **CPU-focused device handling**
   - Force `torch.device("cpu")` for inference.
   - Explicitly load weights/tensors with `map_location="cpu"` where applicable.

3. **Data types & precision**
   - Use `torch.float32` everywhere (diffusers, schedulers, manual tensors).
   - Remove fp16/mixed-precision/autocast code paths.

4. **Diffusers pipeline adjustments**
   - Replace `pipe.to("cuda")` with `pipe.to("cpu")` (or keep on CPU by default).
   - Enable CPU-friendly features: attention slicing, VAE slicing, sequential CPU offload.

5. **CPU performance hygiene**
   - Configure optimal threading via `torch.set_num_threads` and `torch.set_num_interop_threads` using detected cores or configuration values.
   - Wrap inference in `torch.inference_mode()` (or `torch.no_grad()`).
   - Ensure `pipeline.unet`, `vae`, etc., are set to `eval()`.

6. **Environment alignment**
   - Document and honor `OMP_NUM_THREADS` / `MKL_NUM_THREADS` env vars.
   - Docker image must install official CPU wheels via `pip install torch torchvision` (no CUDA extras).

7. **Documentation**
   - Update README, API docs, ARCHITECTURE, implementation plan, and `.env.example` to describe CPU-only deployment, thread tuning, and removal of GPU requirements.

## Secondary Goal (Local Apple Silicon Runner)

Provide one local execution path that can leverage Apple Metal (MPS) when running directly on macOS (not in Docker).

### Deliverables for Local Mode
1. **run_local.py script**
   - Detect MPS via `torch.backends.mps.is_available()`.
   - Prefer `device="mps"` when available; otherwise fall back to CPU.
   - Expose simple CLI/entry point to start the FastAPI app with local settings (thread counts, model dir, etc.).

2. **Configuration hooks**
   - Allow overriding device to `mps` via env vars/CLI while keeping Docker default CPU.
   - Document how to run: `python run_local.py --port 8141 --model-dir ./docker/dev/ai/external/_Models/Stable-diffusion`.

3. **Performance considerations**
   - When on MPS, keep float32 (no fp16) per Apple guidance unless project proves otherwise.
   - Clearly separate Docker CPU instructions from local MPS instructions in docs.

## Quality Gates

- Architecture must remain modular; no business logic inside routes.
- Tests must be updated to reflect CPU defaults and new device reporting (CPU/MPS only).
- Author attribution stays "Inventions4All - github:TWeb79" in any touched files.
- No references to CUDA, ROCm, or discrete GPUs anywhere in the final diff.

## Reporting

When the work is complete, include:
1. Summary of code changes tied to CPU enforcement and local runner implementation.
2. Performance-related justifications (thread counts, inference_mode, slicing, etc.).
3. Instructions for running Docker CPU mode and local MPS mode.