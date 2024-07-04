from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
from PIL import Image
import torch
from io import BytesIO
import base64

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str

def load_model():
    model = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    if torch.cuda.is_available():
        model.to("cuda")
    model.enable_model_cpu_offload()
    return model

model = load_model()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt = request.prompt
    try:
        result = model(prompt=prompt)
        image = result.images[0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
