import os
import sys
import json
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms

# --------------------------------------------------
# Project root
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models.image.alexnet.model import load_alexnet

# --------------------------------------------------
# CONFIG (STRICTEMENT IDENTIQUE À alexnet_predictions)
# --------------------------------------------------
DATA_DIR = "data/images/processed"
CLASSES = ["fake", "real"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = load_alexnet(device=DEVICE)
model.eval()

cases = {"TP": [], "FP": [], "FN": [], "TN": []}

# --------------------------------------------------
# LOOP IDENTIQUE À alexnet_predictions
# --------------------------------------------------
for gt_idx, gt_name in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, gt_name)

    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(folder, filename)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())

        if gt_idx == 1 and pred_idx == 1:
            cases["TP"].append(path)
        elif gt_idx == 0 and pred_idx == 1:
            cases["FP"].append(path)
        elif gt_idx == 1 and pred_idx == 0:
            cases["FN"].append(path)
        elif gt_idx == 0 and pred_idx == 0:
            cases["TN"].append(path)

# --------------------------------------------------
# REPORT
# --------------------------------------------------
print("\nConfusion cases found:")
for k, v in cases.items():
    print(f"{k}: {len(v)}")

selected = {k: v[0] for k, v in cases.items() if len(v) > 0}

print("\nSelected examples:")
for k, v in selected.items():
    print(f"{k}: {v}")

# --------------------------------------------------
# SAVE
# --------------------------------------------------
OUT_JSON = "data/samples/gradcam_cases_alexnet.json"
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

with open(OUT_JSON, "w") as f:
    json.dump(selected, f, indent=2)

print(f"\nSaved -> {OUT_JSON}")
