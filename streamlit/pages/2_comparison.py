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
# Utils – Ground truth & confusion case
# ==================================================
def infer_ground_truth(filename: str, input_type: str):
    name = filename.lower()

    if input_type == "audio":
        if "audio_fake" in name or "fake" in name:
            return 0
        elif "audio_real" in name or "real" in name:
            return 1

    elif input_type == "image":
        if "no_disease" in name or name.startswith("no"):
            return 0
        else:
            return 1

    return None


def confusion_case(y_true, y_pred):
    if y_true is None:
        return "Unknown"
    if y_true == 1 and y_pred == 1:
        return "True Positive"
    if y_true == 0 and y_pred == 0:
        return "True Negative"
    if y_true == 0 and y_pred == 1:
        return "False Positive"
    if y_true == 1 and y_pred == 0:
        return "False Negative"


# ==================================================
# Page config
# ==================================================
st.set_page_config(page_title="XAI Comparison", layout="wide")
st.title("XAI Comparison")
st.write("Side-by-side comparison of explainability methods")

# ==================================================
# Sidebar – Input
# ==================================================
uploaded_file = st.sidebar.file_uploader(
    "Upload an image (.png, .jpg) or audio (.wav)", type=["png", "jpg", "jpeg", "wav"]
)

if uploaded_file is None:
    st.info("Upload a file to start comparison.")
    st.stop()


def detect_input_type(file):
    return "audio" if file.name.lower().endswith(".wav") else "image"


input_type = detect_input_type(uploaded_file)
st.sidebar.markdown(f"**Detected input type:** `{input_type}`")

# ==================================================
# Model selection
# ==================================================
models_available = MODEL_REGISTRY[input_type]

model_key = st.sidebar.selectbox(
    "Select model",
    list(models_available.keys()),
    format_func=lambda k: models_available[k]["name"]
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================
# Preprocessing
# ==================================================
if input_type == "image":
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input image", width=250)
    x, transform = preprocess_image(image, device)

else:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        wav_path = tmp.name

    st.audio(wav_path)
    x, image, transform = preprocess_audio(wav_path, device)
    st.image(image, caption="Spectrogram", width=250)

# ==================================================
# Load model
# ==================================================
model = models_available[model_key]["loader"](device=device)
labels = models_available[model_key]["labels"]

# ==================================================
# Prediction
# ==================================================
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1))
    confidence = probs[0, pred_idx].item()

st.subheader("Prediction")
st.success(
    f"Prediction: **{labels[pred_idx]}** (confidence: {confidence:.3f})"
)

# ==================================================
# Confusion case
# ==================================================
filename = uploaded_file.name
y_true = infer_ground_truth(filename, input_type)
case = confusion_case(y_true, pred_idx)

st.subheader("Prediction analysis")

if case == "True Positive":
    st.success("True Positive – correct detection of a real sample")
elif case == "True Negative":
    st.success("True Negative – correct detection of a fake sample")
elif case == "False Positive":
    st.error("False Positive – fake sample classified as real")
elif case == "False Negative":
    st.error("False Negative – real sample classified as fake")
else:
    st.info("Ground truth could not be inferred from filename")

# ==================================================
# XAI Comparison
# ==================================================
st.divider()
st.subheader("Explainability comparison")

col1, col2, col3 = st.columns(3)

# -------------------------
# Grad-CAM
# -------------------------
with col1:
    st.markdown("### Grad-CAM")
    target_layer = find_last_conv_layer(model)

    if target_layer is None:
        st.warning("Not supported")
    else:
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(x, pred_idx)
        overlay = cam.overlay_on_image(image, heatmap)
        st.image(overlay, use_container_width=True)

# -------------------------
# LIME
# -------------------------
with col2:
    st.markdown("### LIME")
    explainer = LimeExplainer(model, device, transform)
    lime_vis = explainer.explain(np.array(image), pred_idx)
    st.image(lime_vis, use_container_width=True)

# -------------------------
# SHAP (FIXED)
# -------------------------
with col3:
    st.markdown("### SHAP")

    background = torch.zeros_like(x).to(device)
    explainer = ShapExplainer(model, background)
    shap_values = explainer.explain(x)

    # SAME LOGIC AS PAGE 1
    sv = shap_values[0][0]      # (C, H, W)
    heatmap = shap_to_heatmap(sv, target_size=image.size)
    overlay = overlay_shap(image, heatmap)

    st.image(overlay, use_container_width=True)
