from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
from PIL import Image
import torch
from io import BytesIO
import base64
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os
import numpy as np
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# MongoDB client setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27027")  # Make sure the port is correct
MONGO_DB = os.getenv("MONGO_DB", "image_generation_db")

@app.on_event("startup")
async def startup_db_client():
    try:
        app.mongo_client = AsyncIOMotorClient(MONGO_URI)
        app.db_client = app.mongo_client[MONGO_DB]
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection error")

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongo_client.close()
    logger.info("MongoDB connection closed")

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt = request.prompt
    try:
        result = model(prompt=prompt)
        image = result.images[0]

        # Debug: Check for NaN values in the image tensor
        np_image = np.array(image)
        if np.isnan(np_image).any():
            raise ValueError("The generated image contains NaN values.")

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Save image to MongoDB
        image_data = {
            "prompt": prompt,
            "image": img_str
        }
        db_result = await app.db_client.images.insert_one(image_data)
        logger.info(f"Image saved to MongoDB with ID: {db_result.inserted_id}")
        return {"image_id": str(db_result.inserted_id), "image": img_str}
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-image/{image_id}")
async def get_image(image_id: str):
    try:
        image_data = await app.db_client.images.find_one({"_id": ObjectId(image_id)})
        if image_data is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return {"image_id": str(image_data["_id"]), "prompt": image_data["prompt"], "image": image_data["image"]}
    except Exception as e:
        logger.error(f"Error retrieving image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)