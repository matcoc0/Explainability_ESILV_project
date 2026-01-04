import os
import json
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

from models.image.alexnet import load_model

# --------------------------------------------------
# PATHS
# --------------------------------------------------
PROJECT_ROOT = os.getcwd()
CONFIG_PATH = os.path.join(PROJECT_ROOT, "models/image/alexnet/config.yaml")
CASES_PATH = os.path.join(PROJECT_ROOT, "configs/image/alexnet/cases.json")
OUT_BASE = os.path.join(PROJECT_ROOT, "outputs/image/alexnet/lime")

# --------------------------------------------------
# LOAD CONFIG
# --------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

IMG_SIZE = tuple(cfg["input"]["size"])
LABELS = list(cfg["labels"].values())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = load_model(device=DEVICE)
model.eval()

# --------------------------------------------------
# PREDICT FUNCTION (for LIME)
# --------------------------------------------------
def predict_fn(images: np.ndarray):
    """
    images: numpy array (N, H, W, 3) in [0,255]
    returns: probabilities (N, num_classes)
    """
    tensors = []

    for img in images:
        pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        tensors.append(transform(pil))

    batch = torch.stack(tensors).to(DEVICE)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy()

# --------------------------------------------------
# LOAD CASES
# --------------------------------------------------
with open(CASES_PATH, "r") as f:
    cases = json.load(f)

explainer = lime_image.LimeImageExplainer()

print("\nRunning LIME (AlexNet)...\n")

# --------------------------------------------------
# RUN LIME
# --------------------------------------------------
for case_type, paths in cases.items():
    out_dir = os.path.join(OUT_BASE, case_type)
    os.makedirs(out_dir, exist_ok=True)

    for path in paths:
        print(f"[LIME] {case_type} -> {path}")

        image = np.array(Image.open(path).convert("RGB"))

        explanation = explainer.explain_instance(
            image=image,
            classifier_fn=predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=1000
        )

        pred_label = explanation.top_labels[0]

        temp, mask = explanation.get_image_and_mask(
            label=pred_label,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mark_boundaries(temp / 255.0, mask))
        ax.set_title(f"LIME - {case_type} (pred: {LABELS[pred_label]})")
        ax.axis("off")

        out_path = os.path.join(out_dir, os.path.basename(path))
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"Saved -> {out_path}")

print("\nLIME AlexNet completed successfully.")
