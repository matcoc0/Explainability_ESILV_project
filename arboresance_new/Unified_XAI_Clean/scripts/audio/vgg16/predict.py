import os
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.audio.vgg16 import load_model

PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data/audio/spectrograms")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "models/audio/vgg16/config.yaml")

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

y_true, y_pred = [], []

print("\nRunning VGG16 AUDIO predictions...\n")

for gt_idx, gt_name in enumerate(LABELS):
    folder = os.path.join(DATA_DIR, gt_name)

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(folder, fname)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())

        y_true.append(gt_idx)
        y_pred.append(pred_idx)

        print(
            f"{fname:30s} | "
            f"GT={LABELS[gt_idx]:5s} | "
            f"PRED={LABELS[pred_idx]:5s}"
        )

acc = (np.array(y_true) == np.array(y_pred)).mean()
print(f"\nAccuracy (indicative): {acc:.3f}")
