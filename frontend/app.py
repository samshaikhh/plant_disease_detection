"""
app.py (Streamlit frontend)
----------------------------
Simple web UI for the Plant Disease Detection project.

Two modes (in this single file, no separate backend call needed):
1. "Predict" - upload a leaf image, get prediction + remedy
2. "History"  - view past predictions

This version calls model directly (self-contained, good for Streamlit
Cloud deployment where running a separate FastAPI server isn't simple).
If you prefer to call the FastAPI backend instead, see the
`USE_API_BACKEND` flag below.

Run with:
    streamlit run app.py
"""

import json
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent / "backend"))
sys.path.append(str(Path(__file__).parent.parent / "model"))
from database import init_db, save_prediction, get_history  # noqa: E402
from build_model import build_model  # noqa: E402

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
USE_API_BACKEND = False   # set True to call FastAPI at API_URL instead of loading model directly
API_URL = "http://localhost:8000"

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "saved_model" / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "saved_model" / "class_names.json"
REMEDIES_PATH = BASE_DIR / "backend" / "remedies.json"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# VISUAL DESIGN
# ----------------------------------------------------------------------
# NOTE: This section only changes the appearance. The model, prediction,
# database, paths, remedies, and API logic below are intentionally unchanged.
st.markdown(
    """
    <style>
    /* ---------- Top Header / Canvas Integration ---------- */
    header[data-testid="stHeader"] {
        background: linear-gradient(180deg, rgba(14, 43, 31, 0.95) 0%, rgba(18, 56, 40, 0.8) 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        height: 3.8rem !important;
    }

    header[data-testid="stHeader"] * {
        color: #e2f7ed !important;
    }

    /* Adjust main content padding to account for integrated header */
    .main .block-container {
        position: relative;
        z-index: 1;
        max-width: 1050px;
        padding-top: 1.8rem !important;
        padding-bottom: 3rem;
    }

    /* ---------- Global / Botanical Background ---------- */
    .stApp {
        background-color: #f2f9f4;
        background-image: 
            /* Soft botanical leaf vector pattern with 5-8% opacity */
            radial-gradient(circle at 10% 10%, rgba(16, 185, 129, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 90% 90%, rgba(20, 184, 166, 0.08) 0%, transparent 40%),
            url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 100 100" opacity="0.06"><path fill="%23114d33" d="M50 0 C65 20, 90 35, 90 60 C90 80, 70 95, 50 95 C30 95, 10 80, 10 60 C10 35, 35 20, 50 0 Z"/></svg>');
        background-repeat: repeat;
        background-size: auto, auto, 140px 140px;
        min-height: 100vh;
    }

    /* Soft ambient botanical light overlays */
    .stApp::before {
        content: "";
        position: fixed;
        z-index: 0;
        pointer-events: none;
        width: 450px;
        height: 450px;
        left: -120px;
        top: 20%;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(52, 211, 153, 0.14) 0%, rgba(16, 185, 129, 0) 70%);
        filter: blur(40px);
    }

    .stApp::after {
        content: "";
        position: fixed;
        z-index: 0;
        pointer-events: none;
        width: 400px;
        height: 400px;
        right: -100px;
        bottom: 10%;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(20, 184, 166, 0.12) 0%, rgba(13, 148, 136, 0) 70%);
        filter: blur(40px);
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09261a 0%, #0e3827 50%, #124530 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.12);
    }

    [data-testid="stSidebar"] * {
        color: #ecfdf5;
    }

    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 700;
        font-size: .95rem;
        color: #a7f3d0 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: .55rem;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        border-radius: 12px;
        padding: .75rem 1rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all .25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(167, 243, 208, 0.3);
        transform: translateX(3px);
    }

    /* ---------- Premium Hero Card ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.4rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0a3321 0%, #115234 60%, #1a734a 100%);
        box-shadow: 0 20px 40px -15px rgba(10, 51, 33, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .hero:after {
        content: "🌿";
        position: absolute;
        right: 1.2rem;
        top: -0.2rem;
        font-size: 7.5rem;
        opacity: .14;
        transform: rotate(-12deg);
        pointer-events: none;
    }

    .hero h1 {
        color: #ffffff;
        margin: 0 0 .5rem 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -.02em;
    }

    .hero p {
        color: #d1fae5;
        margin: 0;
        max-width: 680px;
        font-size: 1.02rem;
        line-height: 1.65;
        opacity: 0.95;
    }

    /* ---------- Typography & Titles ---------- */
    .section-title {
        margin: 1.6rem 0 .4rem 0;
        color: #0c3825;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .hint {
        color: #526e60;
        font-size: .92rem;
        margin-bottom: 1rem;
        line-height: 1.5;
    }

    .stImage {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(10, 51, 33, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    /* ---------- Modern File Uploader ---------- */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.75);
        border: 2px dashed #6ee7b7;
        border-radius: 20px;
        padding: 0.8rem;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.25s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #10b981;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 14px 30px -5px rgba(16, 185, 129, 0.15);
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
        background: transparent;
    }

    /* ---------- Image Tag / Badge ---------- */
    .image-label {
        display: inline-flex;
        align-items: center;
        margin: 1rem 0 .6rem 0;
        padding: .4rem .85rem;
        border-radius: 999px;
        background: #d1fae5;
        color: #065f46;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .04em;
        border: 1px solid #a7f3d0;
    }

    /* ---------- Result Cards (Glassmorphism) ---------- */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(167, 243, 208, 0.6);
        border-radius: 18px;
        padding: 1.15rem 1.3rem;
        box-shadow: 0 10px 25px -5px rgba(10, 51, 33, 0.06);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    [data-testid="stMetricLabel"] {
        color: #4b6354 !important;
        font-weight: 700;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    [data-testid="stMetricValue"] {
        color: #064e3b !important;
        font-weight: 800;
        font-size: 1.8rem;
    }

    /* Alerts / Remedy Boxes */
    [data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid rgba(16, 185, 129, 0.2);
        box-shadow: 0 8px 20px -4px rgba(10, 51, 33, 0.05);
        backdrop-filter: blur(8px);
    }

    /* ---------- History Table ---------- */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(167, 243, 208, 0.5);
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.04);
        backdrop-filter: blur(10px);
    }

    /* ---------- Primary Buttons ---------- */
    .stButton > button {
        border-radius: 14px;
        font-weight: 700;
        border: 1px solid #a7f3d0;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.55rem 1.25rem;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35);
        border-color: #6ee7b7;
    }

    /* ---------- Footer ---------- */
    .footer {
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(16, 185, 129, 0.15);
        color: #64748b;
        text-align: center;
        font-size: .82rem;
        font-weight: 500;
    }

    /* ---------- Mobile Adjustments ---------- */
    @media (max-width: 700px) {
        header[data-testid="stHeader"] {
            height: 3.2rem !important;
        }

        .main .block-container {
            padding-top: 1.2rem !important;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.5rem;
            border-radius: 18px;
        }

        .hero h1 {
            font-size: 1.75rem;
        }

        .hero:after {
            font-size: 4.5rem;
            right: 0.5rem;
            top: 0.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Branded header
st.markdown(
    """
    <div class="hero">
        <h1>🌿 Plant Disease Detector</h1>
        <p>
            Upload a clear leaf image and let the trained model analyze it,
            identify the predicted disease, and show a suggested remedy.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Cached resource loading (runs once, not on every rerun)
# ----------------------------------------------------------------------
@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        return []
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_model():
    """
    Rebuilds the model architecture locally (from model/build_model.py) and
    loads only the trained weights from the saved .keras file, instead of
    deserializing the full model object. This avoids errors caused by Keras
    version differences between the machine that trained the model (e.g.
    Google Colab) and the machine running this app.
    """
    if not MODEL_PATH.exists():
        return None
    class_names = load_class_names()
    if not class_names:
        return None
    model = build_model(num_classes=len(class_names), img_size=IMG_SIZE)
    model.load_weights(str(MODEL_PATH))
    return model


@st.cache_data
def load_remedies():
    with open(REMEDIES_PATH) as f:
        return json.load(f)


def predict_locally(image: Image.Image, model, class_names, remedies):
    image_resized = image.convert("RGB").resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(image_resized)
    array = np.expand_dims(array, axis=0)
    predictions = model.predict(array)[0]
    idx = int(np.argmax(predictions))
    confidence = float(predictions[idx]) * 100
    predicted_class = class_names[idx] if class_names else str(idx)
    remedy = remedies.get(predicted_class, remedies.get("_default", ""))
    return predicted_class, confidence, remedy


def predict_via_api(image_bytes, filename):
    import requests
    files = {"file": (filename, image_bytes, "image/jpeg")}
    resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["predicted_class"], data["confidence"], data["remedy"]


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
init_db()

st.sidebar.markdown(
    """
    <div style="padding: 1.2rem .4rem 1.1rem .4rem;">
        <div style="font-size:2.2rem;">🌱</div>
        <div style="font-size:1.15rem;font-weight:800;color:#ffffff;">PlantCare AI</div>
        <div style="font-size:.78rem;opacity:.75;margin-top:.2rem;color:#a7f3d0;">
            Plant health assistant
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Navigation", ["Predict", "History"])

if page == "Predict":
    st.markdown('<div class="section-title">🔬 Analyze a leaf</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">For better results, upload a clear JPG or PNG image focused on the leaf.</div>',
        unsafe_allow_html=True,
    )

if page == "Predict":
    st.markdown('<div class="image-label">📤 LEAF IMAGE</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG.",
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        st.markdown('<div class="image-label">🖼️ UPLOADED IMAGE</div>', unsafe_allow_html=True)
        st.image(image, caption="Leaf image", use_column_width=True)

        with st.spinner("Analyzing leaf image..."):
            if USE_API_BACKEND:
                predicted_class, confidence, remedy = predict_via_api(
                    image_bytes, uploaded_file.name
                )
            else:
                model = load_model()
                class_names = load_class_names()
                remedies = load_remedies()

                if model is None:
                    st.error(
                        "Model not found. Please train the model first "
                        "(see model/train.py) before running the app."
                    )
                    st.stop()

                predicted_class, confidence, remedy = predict_locally(
                    image, model, class_names, remedies
                )
                # Save to DB directly since we predicted locally
                save_prediction(uploaded_file.name, predicted_class, confidence, remedy)

        st.success("Analysis complete! 🌱")
        st.markdown('<div class="section-title">📊 Detection Result</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Predicted Disease", predicted_class.replace("___", " - ").replace("_", " "))
        col2.metric("Confidence", f"{confidence:.2f}%")

        st.markdown('<div class="section-title">💡 Suggested Remedy</div>', unsafe_allow_html=True)
        st.info(remedy)

        st.markdown(
            '<div class="footer">AI prediction • Please use the result as a screening aid, not as a laboratory diagnosis.</div>',
            unsafe_allow_html=True,
        )

elif page == "History":
    st.markdown('<div class="section-title">📚 Prediction History</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">Your previous disease predictions are stored here.</div>',
        unsafe_allow_html=True,
    )
    records = get_history(limit=100)

    if not records:
        st.write("No predictions made yet. Go to the Predict page to get started.")
    else:
        # Display as a simple table
        display_rows = [
            {
                "Timestamp": r["timestamp"],
                "Image": r["image_name"],
                "Prediction": r["predicted_class"].replace("___", " - ").replace("_", " "),
                "Confidence (%)": round(r["confidence"], 2),
            }
            for r in records
        ]
        st.dataframe(display_rows, use_container_width=True)