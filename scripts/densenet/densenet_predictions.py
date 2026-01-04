import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.image.densenet.model import load_densenet

DATA_DIR = 'data/images/processed'
CLASSES = ['fake', 'real']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = load_densenet(device=DEVICE)
model.eval()

y_true, y_pred = [], []

print('\nRunning DenseNet predictions...\n')

for label_idx, label_name in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, label_name)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png')))

    for filename in files:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1))

        y_true.append(label_idx)
        y_pred.append(pred)

        print(f'{filename:25s} | GT={label_name} | PRED={CLASSES[pred]} | CONF={probs[0,pred]:.3f}')

acc = (np.array(y_true) == np.array(y_pred)).mean()
print(f'\nAccuracy (indicative): {acc:.3f}')
