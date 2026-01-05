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

from models.audio.mobilenet import load_model

PROJECT_ROOT = os.getcwd()
CONFIG_PATH = os.path.join(PROJECT_ROOT, "models/audio/mobilenet/config.yaml")
CASES_PATH = os.path.join(PROJECT_ROOT, "configs/audio/mobilenet/cases.json")
OUT_BASE = os.path.join(PROJECT_ROOT, "outputs/audio/mobilenet/lime")

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

model = load_model(device=DEVICE)
model.eval()

def predict_fn(images: np.ndarray):
    tensors = []
    for img in images:
        pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        tensors.append(transform(pil))
    batch = torch.stack(tensors).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)

    return probs.cpu().numpy()

with open(CASES_PATH, "r") as f:
    cases = json.load(f)

explainer = lime_image.LimeImageExplainer()

for case_type, paths in cases.items():
    out_dir = os.path.join(OUT_BASE, case_type)
    os.makedirs(out_dir, exist_ok=True)

    for path in paths:
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

        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.axis("off")
        plt.title(f"LIME - {case_type} ({LABELS[pred_label]})")

        out_path = os.path.join(out_dir, os.path.basename(path))
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"[LIME] {case_type} -> {out_path}")
