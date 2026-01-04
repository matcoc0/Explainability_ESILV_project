import os
import sys
import json
import torch
from PIL import Image
from torchvision import transforms

# --------------------------------------------------
# Project root
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.image.densenet.model import load_densenet
from xai.gradcam import GradCAM

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "data/samples/gradcam_densenet"
os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# LOAD CASES (TP / FP / FN / TN)
# --------------------------------------------------
with open("data/samples/gradcam_cases_densenet.json") as f:
    cases = json.load(f)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = load_densenet(device=DEVICE)
model.eval()

# Target layer DenseNet (dernier bloc convolutionnel)
cam = GradCAM(model, target_layer=model.features.denseblock4)

# --------------------------------------------------
# RUN GRAD-CAM
# --------------------------------------------------
for case_name, img_path in cases.items():
    print(f"Processing {case_name}: {img_path}")

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())

    heatmap = cam.generate(x, class_idx=pred)
    overlay = cam.overlay_on_image(img, heatmap)

    out_path = os.path.join(OUT_DIR, f"{case_name}.jpg")
    overlay.save(out_path)

    print(f"{case_name} saved -> {out_path}")
