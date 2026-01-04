import torch
import torch.nn as nn
from torchvision import models


def load_alexnet(device="cpu", num_classes=2, pretrained=True):
    """
    Load AlexNet model for binary classification (fake vs real)
    """
    model = models.alexnet(pretrained=pretrained)

    # Replace classifier
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        num_classes
    )

    model = model.to(device)
    model.eval()
    return model
