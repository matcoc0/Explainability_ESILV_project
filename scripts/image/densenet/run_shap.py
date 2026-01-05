import os
import json
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.image.densenet import load_model
from xai.image.shap import ShapExplainer

PROJECT_ROOT = os.getcwd()
CONFIG_PATH = os.path.join(PROJECT_ROOT, "models/image/densenet/config.yaml")
CASES_PATH = os.path.join(PROJECT_ROOT, "configs/image/densenet/cases.json")
OUT_BASE = os.path.join(PROJECT_ROOT, "outputs/image/densenet/shap")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

IMG_SIZE = tuple(cfg["input"]["size"])
LABELS = list(cfg["labels"].values())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

with open(CASES_PATH, "r") as f:
    cases = json.load(f)

model = load_model(device=DEVICE)
model.eval()

# Build SHAP background
background = []
for paths in cases.values():
    for p in paths:
        background.append(transform(Image.open(p).convert("RGB")))
        if len(background) >= 3:
            break
    if len(background) >= 3:
        break

background = torch.stack(background).to(DEVICE)

explainer = ShapExplainer(model=model, background=background)

for case_type, paths in cases.items():
    out_dir = os.path.join(OUT_BASE, case_type)
    os.makedirs(out_dir, exist_ok=True)

    for path in paths:
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_class = int(torch.argmax(model(x), dim=1).item())

        shap_values = explainer.explain(x)
        sv = shap_values[pred_class][0]
        heatmap = np.mean(np.abs(sv), axis=0)

        plt.imshow(heatmap, cmap="hot")
        plt.axis("off")
        plt.title(f"SHAP - {case_type} ({LABELS[pred_class]})")

        out_path = os.path.join(out_dir, os.path.basename(path))
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"[SHAP] {case_type} -> {out_path}")
