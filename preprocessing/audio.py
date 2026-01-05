# preprocessing/audio.py

import librosa
import numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_audio(wav_path, device="cpu"):
    y, sr = librosa.load(wav_path, sr=16000)

    spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=8000
    )
    spec = librosa.power_to_db(spec, ref=np.max)

    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    spec = (spec * 255).astype(np.uint8)

    img = Image.fromarray(spec).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    x = transform(img).unsqueeze(0).to(device)
    return x, img, transform
