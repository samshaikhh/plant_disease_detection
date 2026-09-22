"""
main.py (FastAPI backend)
--------------------------
REST API for plant disease detection.

Endpoints:
    POST /predict   -> upload an image, get back disease name + confidence + remedy
    GET  /history    -> view past predictions stored in SQLite
    GET  /health      -> simple health check

Run with:
    uvicorn main:app --reload --port 8000
"""

import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from database import init_db, save_prediction, get_history

# ----------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------
app = FastAPI(title="Plant Disease Detection API")

# Allow the Streamlit/React frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # for production, restrict this to your frontend's URL
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR.parent / "saved_model" / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR.parent / "saved_model" / "class_names.json"
REMEDIES_PATH = BASE_DIR / "remedies.json"
IMG_SIZE = (224, 224)

# ----------------------------------------------------------------------
# Load model, class names, remedies ONCE at startup (not per request)
# ----------------------------------------------------------------------
model = None
class_names = []
remedies = {}


@app.on_event("startup")
def load_resources():
    global model, class_names, remedies

    init_db()

    if not MODEL_PATH.exists():
        print(f"WARNING: Model file not found at {MODEL_PATH}. "
              f"Train the model first using model/train.py")
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")

    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
    else:
        print(f"WARNING: class_names.json not found at {CLASS_NAMES_PATH}")

    with open(REMEDIES_PATH) as f:
        remedies = json.load(f)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Converts raw uploaded image bytes into a model-ready numpy array."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)  # add batch dimension
    # NOTE: preprocessing (MobileNetV2 preprocess_input) is baked into the
    # model itself (see model/build_model.py), so we do NOT rescale here.
    return array


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first and restart the server."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    try:
        input_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    predictions = model.predict(input_array)[0]  # shape: (num_classes,)
    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx]) * 100
    predicted_class = class_names[predicted_idx] if class_names else str(predicted_idx)
    remedy = remedies.get(predicted_class, remedies.get("_default", ""))

    # Save this prediction to the database for history tracking
    save_prediction(
        image_name=file.filename,
        predicted_class=predicted_class,
        confidence=confidence,
        remedy=remedy,
    )

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "remedy": remedy,
    }


@app.get("/history")
def history(limit: int = 50):
    return get_history(limit=limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
