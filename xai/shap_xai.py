import torch
import shap
import torch.nn as nn
import numpy as np


# --------------------------------------------------
# Utility: disable inplace ReLU (CRUCIAL for SHAP)
# --------------------------------------------------
def disable_inplace_relu(model):
    """
    SHAP is incompatible with inplace ReLU operations.
    This function disables inplace=True everywhere.
    """
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


# --------------------------------------------------
# SHAP Explainer Wrapper
# --------------------------------------------------
class ShapExplainer:
    def __init__(self, model, background):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Pretrained PyTorch model (AlexNet, DenseNet, etc.)

        background : torch.Tensor
            Background tensor for SHAP
            Shape: (N, C, H, W)
        """
        self.model = model
        self.model.eval()

        # VERY IMPORTANT: disable inplace relu
        disable_inplace_relu(self.model)

        # Ensure background is on same device
        self.device = next(self.model.parameters()).device
        self.background = background.to(self.device)

        # Use GradientExplainer (stable for CNNs)
        self.explainer = shap.GradientExplainer(
            self.model,
            self.background
        )

    def explain(self, x):
        """
        Compute SHAP values for a single input

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            Shape: (1, C, H, W)

        Returns
        -------
        shap_values : list
            SHAP values per class
            Each element shape: (1, C, H, W)
        """
        x = x.to(self.device)

        shap_values = self.explainer.shap_values(x)

        return shap_values


# --------------------------------------------------
# Helper for visualization (optional)
# --------------------------------------------------
def shap_to_heatmap(shap_values):
    """
    Convert SHAP values to a 2D heatmap

    Parameters
    ----------
    shap_values : np.ndarray or torch.Tensor
        Shape: (C, H, W)

    Returns
    -------
    heatmap : np.ndarray
        Shape: (H, W)
    """
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.detach().cpu().numpy()

    # Mean over channels
    heatmap = np.mean(np.abs(shap_values), axis=0)

    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap
