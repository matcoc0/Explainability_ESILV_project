import torch
import shap
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image


# --------------------------------------------------
# Utils
# --------------------------------------------------
def disable_inplace_relu(model):
    """
    Disable inplace ReLU for SHAP compatibility
    """
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


# --------------------------------------------------
# SHAP Explainer
# --------------------------------------------------
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
        Returns:
        --------
        shap_values : list
            shap_values[class] -> (1, C, H, W)
        """
        x = x.to(self.device)
        shap_values = self.explainer.shap_values(x)
        return shap_values


# --------------------------------------------------
# Post-processing
# --------------------------------------------------
def shap_to_heatmap(shap_values, target_size):
    """
    Convert SHAP values (C,H,W) to normalized heatmap (H,W)
    """
    # Tensor -> numpy
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.detach().cpu().numpy()

    # Safety: ensure (C,H,W)
    if shap_values.ndim == 4:
        shap_values = shap_values[0]

    # Mean over channels
    heatmap = np.mean(np.abs(shap_values), axis=0)

    # Resize to original image size
    heatmap = cv2.resize(
        heatmap,
        target_size,
        interpolation=cv2.INTER_LINEAR
    )

    # Normalize to [0,1]
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def overlay_shap(image: Image.Image, heatmap, alpha=0.4):
    """
    Overlay SHAP heatmap on image (SAFE OpenCV version)
    """
    # Convert base image
    img_np = np.array(image).astype(np.uint8)

    # Ensure heatmap is uint8 2D
    heatmap_uint8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)

    # Safety: ensure grayscale
    if heatmap_uint8.ndim != 2:
        raise ValueError("SHAP heatmap must be 2D")

    # Apply colormap
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # Overlay
    overlay = cv2.addWeighted(
        img_np, 1 - alpha,
        heatmap_color, alpha,
        0
    )

    return Image.fromarray(overlay)
