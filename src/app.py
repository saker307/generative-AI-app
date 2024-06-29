from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Any
from PIL import Image
import base64
from io import BytesIO
from model import load_model, generate_image

app = FastAPI()

# Load the model at startup
model = load_model()

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Generative Text-to-Image API"}

@app.post("/generate-image/")
def generate_image_endpoint(input_data: TextInput):
    try:
        image = generate_image(model, input_data.text)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
