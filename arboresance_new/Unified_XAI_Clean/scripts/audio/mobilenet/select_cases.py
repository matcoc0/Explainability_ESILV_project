import os
import json
import yaml
import torch
from PIL import Image
from torchvision import transforms

from models.audio.mobilenet import load_model

PROJECT_ROOT = os.getcwd()

DATA_DIR = os.path.join(PROJECT_ROOT, "data/audio/spectrograms")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "models/audio/mobilenet/config.yaml")
OUT_JSON = os.path.join(PROJECT_ROOT, "configs/audio/mobilenet/cases.json")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

LABELS = list(cfg["labels"].values())
IMG_SIZE = tuple(cfg["input"]["size"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

model = load_model(device=DEVICE)
model.eval()

cases = {"TP": [], "FP": [], "FN": [], "TN": []}

for gt_idx, gt_name in enumerate(LABELS):
    folder = os.path.join(DATA_DIR, gt_name)

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(folder, fname)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_idx = int(torch.argmax(model(x), dim=1).item())

        if gt_idx == 1 and pred_idx == 1:
            cases["TP"].append(path)
        elif gt_idx == 0 and pred_idx == 1:
            cases["FP"].append(path)
        elif gt_idx == 1 and pred_idx == 0:
            cases["FN"].append(path)
        elif gt_idx == 0 and pred_idx == 0:
            cases["TN"].append(path)

selected = {k: v[:1] for k, v in cases.items() if v}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(selected, f, indent=2)

print("Saved cases ->", OUT_JSON)
