import torch
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from io import BytesIO

# ---- CONFIG ----
MODEL_PATH = "./model.safetensors"  # <-- your local file
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- LOAD PIPELINE ----
print("Loading model...")

pipe = StableDiffusionPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float32,   # more stable on Mac
    safety_checker=None
)

pipe = pipe.to(DEVICE)

# ---- SCHEDULER (QUALITY BOOST) ----
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# ---- MAC OPTIMIZATIONS ----
pipe.enable_attention_slicing()          # prevents memory spikes
pipe.enable_sequential_cpu_offload()     # better RAM usage

print(f"Running on: {DEVICE}")

# ---- API ----
app = FastAPI()

@app.get("/generate")
def generate(
    prompt: str = Query(..., description="Main prompt"),
    negative_prompt: str = Query(
        "blurry, bad anatomy, extra fingers, extra limbs, deformed, low quality, worst quality",
        description="Negative prompt"
    ),
    steps: int = 25,
    guidance: float = 6.0,
    width: int = 512,
    height: int = 512,
    seed: int = 0
):
    # ---- SEED CONTROL ----
    generator = torch.Generator(device=DEVICE)
    if seed != 0:
        generator = generator.manual_seed(seed)

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator
        )

    image = result.images[0]

    # ---- RETURN IMAGE ----
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")