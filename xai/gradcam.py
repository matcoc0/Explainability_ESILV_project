# xai/gradcam.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


# ==================================================
# Utils
# ==================================================
def disable_inplace_relu(model: nn.Module):
    """
    Disable inplace ReLU to avoid autograd + backward hook crashes.
    """
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


def find_last_conv_layer(model: nn.Module):
    """
    Find last Conv2d layer (robust across architectures).
    """
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


# ==================================================
# Grad-CAM
# ==================================================
class GradCAM:
    """
    Stable Grad-CAM implementation (PyTorch 2+ safe)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        disable_inplace_relu(self.model)

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)

        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(backward_hook)
        else:
            self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, class_idx: int):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        # cam is still in feature-map space → resize later
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    @staticmethod
    def overlay_on_image(
        img: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ):
        """
        Resize heatmap to image size BEFORE overlay.
        """
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # 🔴 CRITICAL FIX
        heatmap = cv2.resize(heatmap, (w, h))

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(
            img_np,
            1 - alpha,
            heatmap,
            alpha,
            0
        )

        return Image.fromarray(overlay)
