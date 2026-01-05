import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image


class LimeExplainer:
    """
    Wrapper for LIME image explanations
    """

    def __init__(self, model, device, transform):
        self.model = model
        self.device = device
        self.transform = transform
        self.model.eval()

        self.explainer = lime_image.LimeImageExplainer()

    def predict(self, images):
        """
        images: numpy array (N, H, W, 3) in [0,255]
        returns: probabilities (N, num_classes)
        """
        batch = []

        for img in images:
            img = img.astype(np.uint8)
            pil_img = Image.fromarray(img)   # ✅ FIX ICI
            tensor = self.transform(pil_img).unsqueeze(0)
            batch.append(tensor)

        batch = torch.cat(batch).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def explain(self, image_np, class_idx):
        explanation = self.explainer.explain_instance(
            image_np,
            self.predict,
            top_labels=2,
            hide_color=0,
            num_samples=1000
        )

        temp, mask = explanation.get_image_and_mask(
            class_idx,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        return mark_boundaries(temp / 255.0, mask)
