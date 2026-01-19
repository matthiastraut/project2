import onnxruntime as ort
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI
import uvicorn
import requests

app = FastAPI()

# Load the model and create an inference session
session = ort.InferenceSession("model.onnx")

# Get input name (usually 'input')
input_name = session.get_inputs()[0].name

# Get the metadata from the session
meta = session.get_modelmeta().custom_metadata_map
if 'class_names' in meta:
    class_names = json.loads(meta['class_names'])
else:
    print("Warning: Metadata not found in ONNX file. Using generic labels.")
    class_names = [f"Class_{i}" for i in range(70)]

@app.get("/")
def root():
    return {"status": "App is working."}

@app.post("/predict")
async def predict(input_image):

    # Load an image as test
    img = Image.open(requests.get(input_image, stream=True).raw)

    ## Resize to target size
    img = img.resize((300, 200))

    # Convert to numpy array
    x = np.array(img)
    print(x.shape)  # (200, 300, 3)

    # We use ImageNet-like preprocessing
    input_size = (300, 200)

    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Simple transforms - just resize and normalize
    eval_transforms = transforms.Compose([
        transforms.Resize((input_size[0], input_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Apply transforms (Resize, ToTensor, Normalize)
    # Result is a torch.Tensor of shape (3, 200, 300) with float32 values
    img_tensor = eval_transforms(img)

    # Add batch dimension: (3, 200, 300) -> (1, 3, 200, 300)
    img_tensor = img_tensor.unsqueeze(0)

    # Convert to numpy array (ONNX Runtime requires numpy)
    x = img_tensor.numpy().astype(np.float32)

    # Run the model
    outputs = session.run(None, {input_name: x})

    # print(dict(zip(class_names, outputs[0][0].tolist())))
    # Use it to decode the prediction
    predicted_idx = np.argmax(outputs[0])
    #print(f"Predicted Aircraft: {class_names[predicted_idx]}")
    # print(class_names)

    return {
        #"Identified Aircraft:": class_names[predicted_idx]
        "Entire output": dict(zip(class_names, outputs[0][0].tolist()))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
