import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# --------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.image.alexnet.model import load_alexnet

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["fake", "real"]

CASES_JSON = "data/samples/gradcam_cases_alexnet.json"
OUT_DIR = "data/samples/lime_alexnet"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 224  # IMPORTANT: PAS UN TUPLE

# --------------------------------------------------
# TRANSFORM (IDENTIQUE À ALEXNET_PREDICTIONS)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

# --------------------------------------------------
# PREDICTION FUNCTION FOR LIME
# --------------------------------------------------
def predict_fn(images: np.ndarray):
    """
    images: numpy array (N, H, W, 3) in [0,255]
    returns: probabilities (N, 2)
    """
    batch = []

    for img in images:
        pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        x = transform(pil)
        batch.append(x)

    batch = torch.stack(batch).to(DEVICE)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy()

# --------------------------------------------------
# LOAD SELECTED CASES (TP / FP / FN / TN)
# --------------------------------------------------
with open(CASES_JSON, "r") as f:
    cases = json.load(f)

print("\nRunning LIME on selected cases...\n")

# --------------------------------------------------
# LIME EXPLAINER
# --------------------------------------------------
explainer = lime_image.LimeImageExplainer()

for case_name, img_path in cases.items():
    print(f"Processing {case_name}: {img_path}")

    image = np.array(Image.open(img_path).convert("RGB"))

    explanation = explainer.explain_instance(
        image=image,
        classifier_fn=predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=1000
    )

    # Explain predicted class
    pred_label = explanation.top_labels[0]

    temp, mask = explanation.get_image_and_mask(
        label=pred_label,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    # Visualization
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mark_boundaries(temp / 255.0, mask))
    ax.set_title(f"LIME - {case_name} (pred: {CLASSES[pred_label]})")
    ax.axis("off")

    out_path = os.path.join(OUT_DIR, f"{case_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {out_path}")

print("\nLIME AlexNet finished successfully.")
