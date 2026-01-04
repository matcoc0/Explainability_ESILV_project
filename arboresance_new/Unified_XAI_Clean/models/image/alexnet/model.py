import yaml
import torch
import os
import torch.nn as nn
from torchvision import models

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(device="cpu"):
    with open(os.path.join(THIS_DIR, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["num_classes"]
    pretrained = cfg.get("pretrained", False)

    model = models.alexnet(pretrained=pretrained)
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        num_classes
    )

    model.to(device)
    model.eval()
    return model
