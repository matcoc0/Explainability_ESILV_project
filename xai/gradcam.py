import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)

        score = output[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def overlay_on_image(self, img: Image.Image, heatmap, alpha=0.4):
        img = np.array(img)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        return Image.fromarray(overlay)
