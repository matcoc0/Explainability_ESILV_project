# models/image/densenet/model.py

import os
import yaml
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(device="cpu"):
    with open(os.path.join(THIS_DIR, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["num_classes"]

    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes
    )

    model.to(device)
    model.eval()
    return model
