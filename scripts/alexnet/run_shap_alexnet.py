import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.image.alexnet.model import load_alexnet
from xai.shap_xai import ShapExplainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "data/samples/shap_alexnet"
os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load cases (TP / FP / FN / TN)
# -----------------------------
with open("data/samples/gradcam_cases_alexnet.json") as f:
    cases = json.load(f)

# -----------------------------
# Load model
# -----------------------------
model = load_alexnet(device=DEVICE)
model.eval()

# -----------------------------
# Background (IMPORTANT)
# -----------------------------
background = []
for path in list(cases.values())[:3]:
    img = Image.open(path).convert("RGB")
    background.append(transform(img))
background = torch.stack(background).to(DEVICE)

explainer = ShapExplainer(model, background)

# -----------------------------
# Run SHAP
# -----------------------------
for name, path in cases.items():
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    shap_values = explainer.explain(x)
    sv = shap_values[0][0]  # (C,H,W)

    heatmap = np.mean(np.abs(sv), axis=0)

    plt.imshow(heatmap, cmap="hot")
    plt.axis("off")
    plt.title(f"SHAP - {name}")

    out_path = os.path.join(OUT_DIR, f"{name}.jpg")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"{name} saved -> {out_path}")
