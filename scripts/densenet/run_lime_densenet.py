import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.image.densenet.model import load_densenet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "data/samples/lime_densenet"
os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

with open("data/samples/gradcam_cases_densenet.json") as f:
    cases = json.load(f)

model = load_densenet(device=DEVICE)
model.eval()

def predict_fn(images):
    tensors = []
    for img in images:
        pil = Image.fromarray(img.astype("uint8"))
        tensors.append(transform(pil))
    batch = torch.stack(tensors).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)
    return probs.cpu().numpy()

explainer = lime_image.LimeImageExplainer()

for name, path in cases.items():
    image = np.array(Image.open(path).convert("RGB"))

    explanation = explainer.explain_instance(
        image,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    plt.imshow(mark_boundaries(temp, mask))
    plt.axis("off")
    plt.title(f"LIME - {name}")
    plt.savefig(os.path.join(OUT_DIR, f"{name}.png"), bbox_inches="tight")
    plt.close()

    print(f"LIME {name} saved")
