from torch.utils.data import Dataset
import subprocess
import numpy as np
import random
import torch
import librosa

SR = 16000
SEG_SEC = 1.2
SEG_SAMPLES = int(SEG_SEC * SR)


def crop_center(y: np.ndarray, center: float, seg_len: int) -> np.ndarray:
    if len(y) < seg_len:
        return np.pad(y, (0, seg_len - len(y)))
    start = int(center - seg_len // 2)
    start = max(0, min(start, len(y) - seg_len))
    return y[start:start+seg_len]

def random_crop(y: np.ndarray, seg_len: int) -> np.ndarray:
    if len(y) < seg_len:
        return np.pad(y, (0, seg_len - len(y)))
    start = random.randint(0, len(y) - seg_len)
    return y[start:start+seg_len]

def load_opus_ffmpeg(path: str, sr: int = SR) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sr),
        "pipe:1"
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    y = np.frombuffer(p.stdout, dtype=np.float32)
    return y

N_FFT = 1024
HOP   = 160
WIN   = 400
N_MELS = 80
FMIN = 20
FMAX = 8000

def logmel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window="hann",
        center=True,
        power=2.0,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=min(FMAX, sr/2),
        htk=True,
        norm=None
    )
    return np.log(S + 1e-10).astype(np.float32)   # [n_mels, T]


class KWSLazyDataset(Dataset):
    def __init__(self, examples, sr=SR, seg_samples=SEG_SAMPLES):
        self.examples = examples
        self.sr = sr
        self.seg_samples = seg_samples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        path, label, s, e = self.examples[idx]

        y = load_opus_ffmpeg(path, sr=self.sr)

        if label == 1:
            center = 0.5 * (s + e) * self.sr
            seg = crop_center(y, center, self.seg_samples)
        else:
            seg = random_crop(y, self.seg_samples)

        feat = logmel(seg, sr=self.sr)                  # [80, T]
        x = torch.from_numpy(feat).unsqueeze(0)         # [1, 80, T]
        yb = torch.tensor(label, dtype=torch.float32)
        return x, yb
