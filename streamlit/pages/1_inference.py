# ==================================================
# FORCE PROJECT ROOT
# ==================================================
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==================================================
# Imports
# ==================================================
import streamlit as st
import torch
import numpy as np
from PIL import Image
import tempfile

from models.registry import MODEL_REGISTRY
from preprocessing.image import preprocess_image
from preprocessing.audio import preprocess_audio

from xai.gradcam import GradCAM, find_last_conv_layer
from xai.lime import LimeExplainer
from xai.shap import ShapExplainer, shap_to_heatmap, overlay_shap

# ==================================================
# Page UI
# ==================================================
st.header("Inference & Explainability")

uploaded_file = st.sidebar.file_uploader(
    "Upload an image (.png, .jpg) or audio (.wav)", type=["png", "jpg", "jpeg", "wav"]
)

def detect_input_type(file):
    return "audio" if file.name.lower().endswith(".wav") else "image"

if uploaded_file is None:
    st.info("Upload an image or a wav file to start.")
    st.stop()

# ==================================================
# Selection
# ==================================================
input_type = detect_input_type(uploaded_file)
st.sidebar.markdown(f"**Detected input:** `{input_type}`")

models_available = MODEL_REGISTRY[input_type]

model_key = st.sidebar.selectbox(
    "Select model", list(models_available.keys()), format_func=lambda k: models_available[k]["name"]
)

xai_method = st.sidebar.selectbox("Select XAI method", ["gradcam", "lime", "shap"])

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================
# Preprocessing
# ==================================================
if input_type == "image":
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input image", width=300)
    x, transform = preprocess_image(image, device)

else:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        wav_path = tmp.name

    st.audio(wav_path)
    x, image, transform = preprocess_audio(wav_path, device)
    st.image(image, caption="Spectrogram", width=300)

# ==================================================
# Model & prediction
# ==================================================
model = models_available[model_key]["loader"](device=device)
labels = models_available[model_key]["labels"]

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1))
    confidence = probs[0, pred_idx].item()

st.success(
    f"Prediction: **{labels[pred_idx]}** (confidence: {confidence:.3f})"
)

# ==================================================
# Explainability
# ==================================================
st.divider()
st.subheader("Explainability")

if xai_method == "gradcam":
    target_layer = find_last_conv_layer(model)
    if target_layer is None:
        st.warning("Grad-CAM not supported.")
    else:
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(x, pred_idx)
        overlay = cam.overlay_on_image(image, heatmap)
        st.image(overlay, caption="Grad-CAM")

elif xai_method == "lime":
    explainer = LimeExplainer(model, device, transform)
    lime_vis = explainer.explain(np.array(image), pred_idx)
    st.image(lime_vis, caption="LIME")

elif xai_method == "shap":
    background = torch.zeros_like(x).to(device)
    explainer = ShapExplainer(model, background)
    shap_values = explainer.explain(x)

    heatmap = shap_to_heatmap(
        shap_values[0],  # robuste même si 1 seule classe
        target_size=image.size
    )
    overlay = overlay_shap(image, heatmap)
    st.image(overlay, caption="SHAP")
