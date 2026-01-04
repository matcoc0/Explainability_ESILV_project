import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

def load_densenet(device="cpu"):
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model = model.to(device)
    return model
