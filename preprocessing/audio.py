'''
import librosa
import numpy as np
from PIL import Image
from torchvision import transforms


def preprocess_audio(wav_path, device="cpu"):
    y, sr = librosa.load(wav_path, sr=16000)

    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spec = librosa.power_to_db(spec, ref=np.max)

    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    spec = (spec * 255).astype(np.uint8)

    img = Image.fromarray(spec).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    x = transform(img).unsqueeze(0).to(device)
    return x, img, transform
'''
import librosa
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.cm as cm


def preprocess_audio(
    wav_path,
    device="cpu",
    target_sr=16000,
    n_mels=128,
    n_fft=1024,
    hop_length=256,
    fmin=20,
    fmax=8000
):
    # load audio
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)

    # trim silence (important)
    y, _ = librosa.effects.trim(y, top_db=30)

    # mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )

    # log scale
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # clip dynamic range (visual stability)
    mel_db = np.clip(mel_db, -80, 0)

    # normalize to [0, 1]
    mel_norm = (mel_db + 80) / 80

    # apply matplotlib colormap 
    cmap = cm.get_cmap("magma")   # same visual family as librosa
    mel_rgb = cmap(mel_norm)[:, :, :3]  # drop alpha
    mel_rgb = (mel_rgb * 255).astype(np.uint8)

    # Flip frequency axis (librosa-style)
    mel_rgb = np.flipud(mel_rgb)

    # PIL Image
    img = Image.fromarray(mel_rgb)

    # CNN preprocessing (ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    x = transform(img).unsqueeze(0).to(device)

    return x, img, transform
