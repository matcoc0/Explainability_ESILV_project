import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.image.alexnet import load_model

PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data/images/processed")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "models/image/alexnet/config.yaml")

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

y_true, y_pred = [], []

print("\nRunning AlexNet predictions...\n")

for gt_idx, gt_name in enumerate(LABELS):
    folder = os.path.join(DATA_DIR, gt_name)
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )

    for filename in files:
        path = os.path.join(folder, filename)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            conf = probs[0, pred_idx].item()

        y_true.append(gt_idx)
        y_pred.append(pred_idx)

        print(
            f"{filename:25s} | "
            f"GT={LABELS[gt_idx]:5s} | "
            f"PRED={LABELS[pred_idx]:5s} | "
            f"CONF={conf:.3f}"
        )

# --------------------------------------------------
# METRICS
# --------------------------------------------------
accuracy = (np.array(y_true) == np.array(y_pred)).mean()

print("\nSummary")
print("-------")
print(f"Total samples : {len(y_true)}")
print(f"Accuracy      : {accuracy:.3f}")
