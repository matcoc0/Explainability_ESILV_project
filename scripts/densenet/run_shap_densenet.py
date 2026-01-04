import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.image.densenet.model import load_densenet
from xai.shap_xai import ShapExplainer, shap_to_heatmap

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_JSON = "data/samples/gradcam_cases_densenet.json"
OUT_DIR = "data/samples/shap_densenet"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)

# --------------------------------------------------
# TRANSFORM (IDENTIQUE AUX PRÉDICTIONS)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# LOAD CASES
# --------------------------------------------------
with open(DATA_JSON, "r") as f:
    cases = json.load(f)

print("\nRunning SHAP on DenseNet cases...\n")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = load_densenet(device=DEVICE)
model.eval()

# --------------------------------------------------
# BUILD BACKGROUND (IMPORTANT)
# --------------------------------------------------
background_tensors = []

for _, img_path in list(cases.items())[:2]:
    img = Image.open(img_path).convert("RGB")
    x = transform(img)
    background_tensors.append(x)

background = torch.stack(background_tensors).to(DEVICE)

# --------------------------------------------------
# INIT SHAP
# --------------------------------------------------
explainer = ShapExplainer(
    model=model,
    background=background
)

# --------------------------------------------------
# RUN SHAP
# --------------------------------------------------
for case_name, img_path in cases.items():
    print(f"Processing {case_name}: {img_path}")

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred_class = int(torch.argmax(logits, dim=1).item())

    shap_values = explainer.explain(x)

    # SHAP values for predicted class
    shap_map = shap_to_heatmap(shap_values[pred_class][0])

    # Resize heatmap
    heatmap = cv2.resize(shap_map, IMG_SIZE)

    # Overlay
    img_np = np.array(img.resize(IMG_SIZE))
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        img_np,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    out_path = os.path.join(OUT_DIR, f"{case_name}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Saved -> {out_path}")

print("\nSHAP DenseNet completed successfully.")
