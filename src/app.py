from flask import Flask, request, jsonify
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the diffusion model pipeline
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()

@app.route('/generate', methods=['GET'])
def generate_image():
    # Get the prompt from the query parameters
    prompt = request.args.get(
        'prompt', 
        default="underwater scene pendant design. All elements should be in a sterling silver look, with no holes in the design. The design should be suitable for production with a CNC milling machine or a CNC laser.",
        type=str
    )

    # Generate the image using the model
    generated_image = pipe(prompt=prompt).images[0]
    
    # Convert the PIL image to a byte array
    img_byte_arr = io.BytesIO()
    generated_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Convert the byte array to a base64 string
    img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')

    # Return the image as a base64 string in the JSON response
    return jsonify({'image': img_base64})

if __name__== '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

