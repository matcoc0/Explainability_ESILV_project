import torch
import shap
import torch.nn as nn
import numpy as np


def disable_inplace_relu(model):
    """
    Disable inplace ReLU for SHAP compatibility
    """
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


class ShapExplainer:
    """
    SHAP GradientExplainer wrapper for CNNs
    """

    def __init__(self, model, background):
        self.model = model
        self.model.eval()

        disable_inplace_relu(self.model)

        self.device = next(self.model.parameters()).device
        self.background = background.to(self.device)

        self.explainer = shap.GradientExplainer(
            self.model,
            self.background
        )

    def explain(self, x):
        """
        Compute SHAP values for a single input

        x : torch.Tensor (1, C, H, W)
        """
        x = x.to(self.device)
        shap_values = self.explainer.shap_values(x)
        return shap_values


def shap_to_heatmap(shap_values):
    """
    Convert SHAP values (C,H,W) to heatmap (H,W)
    """
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.detach().cpu().numpy()

    heatmap = np.mean(np.abs(shap_values), axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap
