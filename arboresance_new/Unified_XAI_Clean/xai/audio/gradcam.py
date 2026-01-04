import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Standard Grad-CAM implementation for CNNs
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x, class_idx):
        """
        Generate Grad-CAM heatmap

        Parameters
        ----------
        x : torch.Tensor
            Shape (1, C, H, W)
        class_idx : int
            Target class index
        """
        self.model.zero_grad()

        logits = self.model(x)
        score = logits[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    @staticmethod
    def overlay_on_image(img: Image.Image, heatmap, alpha=0.4):
        img_np = np.array(img)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
        return Image.fromarray(overlay)
