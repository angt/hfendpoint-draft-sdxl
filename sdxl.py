import base64
import io
import uuid
from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline
import hfendpoint

try:
    from diffusers.utils.logging import disable_progress_bar
    disable_progress_bar()
except ImportError:
    print("Could not disable diffusers progress bar.")

if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
    variant = "fp16"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float16
    variant = "fp16"
else:
    device = "cpu"
    torch_dtype = torch.float32
    variant = None

print(f"Using device: {device}")

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch_dtype,
    use_safetensors=True,
    variant=variant
)
pipe.to(device)

pipe_edit = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch_dtype,
    use_safetensors=True,
    variant=variant
)
pipe_edit.to(device)

def stable_diffusion_handler(payload, send_chunk):
    prompt = payload["prompt"]
    n = payload.get("n", 1)
    width = payload.get("width", 1024)
    height = payload.get("height", 1024)
    response_format = payload.get("response_format", "b64_json")

    result = pipe(
        prompt,
        num_images_per_prompt=n,
        width=width,
        height=height
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    for image in result.images:
        image_data = {}
        if response_format == "b64_json":
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_data["b64_json"] = b64_data
        else:
            filename = f"{uuid.uuid4()}.png"
            file_path = output_dir / filename
            image.save(file_path)
            image_data["url"] = str(file_path)
        send_chunk(image_data)

def stable_diffusion_edit_handler(payload, send_chunk):
    prompt = payload["prompt"]
    n = payload.get("n", 1)
    width = payload.get("width", 1024)
    height = payload.get("height", 1024)
    response_format = payload.get("response_format", "b64_json")
    image_bytes = payload.get("image")
    mask_bytes = payload.get("mask")

    init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    init_image = init_image.resize((width, height))

    mask_image = None
    if mask_bytes:
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert("RGB")
        mask_image = mask_image.resize((width, height))

    result = pipe_edit(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        num_images_per_prompt=n,
        width=width,
        height=height,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    for image in result.images:
        image_data = {}
        if response_format == "b64_json":
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_data["b64_json"] = b64_data
        else:
            filename = f"{uuid.uuid4()}.png"
            file_path = output_dir / filename
            image.save(file_path)
            image_data["url"] = str(file_path)
        send_chunk(image_data)

if __name__ == "__main__":
    hfendpoint.run({
        "images_generations": stable_diffusion_handler,
        "images_editions": stable_diffusion_edit_handler
    })
