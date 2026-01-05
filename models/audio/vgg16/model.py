import os
import yaml
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(device="cpu"):
    """
    Load VGG16 for audio spectrogram classification.

    - Uses ImageNet pretrained weights by default
    - Optionally loads custom weights.pth if present
    """

    with open(os.path.join(THIS_DIR, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["num_classes"]
    pretrained = cfg.get("pretrained", True)

    model = models.vgg16(
        weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    )

    model.classifier[6] = nn.Linear(4096, num_classes)

    weights_path = os.path.join(THIS_DIR, "weights.pth")
    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
        print("[INFO] Loading custom VGG16 audio weights")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("[INFO] Using ImageNet pretrained weights")

    model.to(device)
    model.eval()
    return model
