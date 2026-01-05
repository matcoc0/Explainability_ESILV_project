import os
import yaml
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(device="cpu"):
    """
    Load MobileNetV2 for audio spectrogram classification.
    Uses ImageNet pretrained weights by default.
    Custom weights are OPTIONAL (not required).
    """

    with open(os.path.join(THIS_DIR, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["num_classes"]
    pretrained = cfg.get("pretrained", True)


    model = models.mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT if pretrained else None
    )

    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

 
    weights_path = os.path.join(THIS_DIR, "weights.pth")

    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
        print("[INFO] Loading custom MobileNet audio weights")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("[INFO] Using ImageNet pretrained weights (no custom weights)")

  
    model.to(device)
    model.eval()
    return model
