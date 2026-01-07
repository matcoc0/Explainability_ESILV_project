import os
import json
import yaml
import torch
from PIL import Image
from torchvision import transforms

from models.image.densenet import load_model
from xai.gradcam import GradCAM

PROJECT_ROOT = os.getcwd()

CONFIG_PATH = os.path.join(PROJECT_ROOT, "models/image/densenet/config.yaml")
CASES_PATH = os.path.join(PROJECT_ROOT, "configs/image/densenet/cases.json")
OUT_BASE = os.path.join(PROJECT_ROOT, "outputs/image/densenet/gradcam")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

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

with open(CASES_PATH, "r") as f:
    cases = json.load(f)

model = load_model(device=DEVICE)
model.eval()

cam = GradCAM(model, target_layer=model.features.denseblock4)

for case_type, paths in cases.items():
    out_dir = os.path.join(OUT_BASE, case_type)
    os.makedirs(out_dir, exist_ok=True)

    for path in paths:
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = int(torch.argmax(model(x), dim=1).item())

        heatmap = cam.generate(x, class_idx=pred)
        overlay = cam.overlay_on_image(img, heatmap)

        out_path = os.path.join(out_dir, os.path.basename(path))
        overlay.save(out_path)

        print(f"[GradCAM] {case_type} -> {out_path}")
