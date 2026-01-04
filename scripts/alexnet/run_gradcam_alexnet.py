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
from xai.gradcam import GradCAM

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CASES_FILE = "data/samples/gradcam_cases_alexnet.json"
OUT_DIR = "data/samples/gradcam_alexnet"
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
# LOAD CASES
# --------------------------------------------------
with open(CASES_FILE, "r") as f:
    cases = json.load(f)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = load_alexnet(device=DEVICE)
model.eval()

cam = GradCAM(model, target_layer=model.features[12])

# --------------------------------------------------
# RUN GRADCAM
# --------------------------------------------------
for case_name, img_path in cases.items():
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = int(torch.argmax(probs, dim=1).item())

    heatmap = cam.generate(x, class_idx=pred)
    overlay = cam.overlay_on_image(img, heatmap)

    out_path = os.path.join(OUT_DIR, f"{case_name}.jpg")
    overlay.save(out_path)

    print(f"{case_name} -> {out_path}")
