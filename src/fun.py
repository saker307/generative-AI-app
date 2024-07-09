from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import io
import base64

app = FastAPI()

# Load the Stable Diffusion model
model_id = "stabilityai/stable-diffusion-2-1"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

class TextInput(BaseModel):
    prompt: str

@app.post("/generate-image/")
async def generate_image(input: TextInput):
    try:
        # Generate the image
        image = pipeline(prompt=input.prompt).images[0]
        
        # Convert the PIL image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        img_bytes = buffer.read()
        
        # Encode the image to base64
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        
        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)