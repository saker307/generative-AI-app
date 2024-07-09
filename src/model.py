from diffusers import DiffusionPipeline
import torch
from PIL import Image
from typing import Any

def load_model():
    model = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    model.to("cpu")
    #model.enable_model_cpu_offload()
    return model

def generate_image(model: Any, text: str) -> Image:
    result = model(prompt=text)
    image = result.images[0]
    return image