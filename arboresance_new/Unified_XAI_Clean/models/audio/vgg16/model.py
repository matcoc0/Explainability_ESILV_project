# models/audio/vgg16/model.py

import os
import yaml
import torch
import torch.nn as nn
from torchvision import models

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(device="cpu"):
    with open(os.path.join(THIS_DIR, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["num_classes"]

    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, num_classes)

    weights_path = os.path.join(THIS_DIR, "weights.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights.pth not found in {THIS_DIR}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
