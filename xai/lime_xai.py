# xai/lime_xai.py
import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries

class LimeImageExplainerWrapper:
    def __init__(self, model, device, transform):
        self.model = model
        self.device = device
        self.transform = transform
        self.explainer = lime_image.LimeImageExplainer()

    def predict(self, images):
        """
        images: numpy array [N, H, W, 3] in [0,255]
        """
        self.model.eval()
        batch = []

        for img in images:
            img = img.astype(np.uint8)
            img_t = self.transform(img).unsqueeze(0)
            batch.append(img_t)

        batch = torch.cat(batch).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def explain(self, image_np, label):
        explanation = self.explainer.explain_instance(
            image_np,
            self.predict,
            top_labels=2,
            hide_color=0,
            num_samples=1000
        )

        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )

        return mark_boundaries(temp / 255.0, mask)
