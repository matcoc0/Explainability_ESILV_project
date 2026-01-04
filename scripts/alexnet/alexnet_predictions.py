import os
import sys

# -------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.image.alexnet.model import load_alexnet

# --------------------
# CONFIG
# --------------------
DATA_DIR = "data/images/processed"
CLASSES = ["fake", "real"]
IMG_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# TRANSFORM
# --------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# LOAD MODEL
# --------------------
model = load_alexnet(device=DEVICE)
model.eval()

# --------------------
# PREDICTION LOOP
# --------------------
y_true = []
y_pred = []

print("\nRunning AlexNet predictions...\n")

for label_idx, label_name in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, label_name)
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"Processing {label_name.upper()} ({len(files)} images)\n")

    for filename in files:
        path = os.path.join(folder, filename)

        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()

        y_true.append(label_idx)
        y_pred.append(pred)

        print(
            f"{filename:30s} | "
            f"GT: {label_name:5s} | "
            f"PRED: {CLASSES[pred]:5s} | "
            f"CONF: {confidence:.3f}"
        )

    print("-" * 70)

# --------------------
# METRICS
# --------------------
y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = (y_true == y_pred).mean()

print("\nSummary")
print("-------")
print(f"Total samples : {len(y_true)}")
print(f"Accuracy      : {accuracy:.3f}")
