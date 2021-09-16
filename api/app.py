"""A local server for testing model predictions."""

import json
from pathlib import Path
import time

import cv2
from fastapi import FastAPI, File, UploadFile
import numpy as np
import torch
import torch.jit

MODEL_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = MODEL_FOLDER / "model.pt"
MODEL_DESC_PATH = MODEL_FOLDER / "model_desc.json"

app = FastAPI()


@app.get("/")
async def root():
    """Provides simple check route."""
    return "Parking Spot Detection API."


@app.post("/predict")
async def predict(image_file: UploadFile = File(...)):
    """Provides API route for prediction. Expects an image."""
    start_time = time.perf_counter()
    image_buffer = await image_file.read()
    image = cv2.imdecode(np.frombuffer(image_buffer, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(torch.float32)
    mean = model_desc["preprocessing"]["normalize"]["mean"]
    mean = torch.tensor(mean).reshape(1, -1, 1, 1)  # pylint: disable=not-callable
    std = model_desc["preprocessing"]["normalize"]["std"]
    std = torch.tensor(std).reshape(1, -1, 1, 1)  # pylint: disable=not-callable
    image_tensor = (image_tensor / 255.0 - mean) / std

    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = output.argmax().item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0, predicted_class].item()
    class_string = model_desc["mapping"][str(predicted_class)]
    execution_time = time.perf_counter() - start_time

    return {"class": class_string, "confidence": confidence, "execution_time": execution_time}


def load_model():
    """Loads and returns torchscript model."""
    return torch.jit.load(str(MODEL_PATH))


def load_description():
    """Loads and returns model description."""
    with MODEL_DESC_PATH.open("r") as f:
        desc = json.load(f)
    return desc


model = load_model()
model_desc = load_description()
