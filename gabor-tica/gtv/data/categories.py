"""Category data for Phase 5+: Fashion-MNIST (10 clothing categories).

Files are downloaded once into ``data_cache/fashion_mnist`` (git-ignored) from
the dataset's GitHub repository. Images are placed, upsampled 2x, in the
center of a ``size`` x ``size`` canvas and made zero-mean per image.
"""

import gzip
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CACHE = Path(__file__).resolve().parents[2] / "data_cache" / "fashion_mnist"
URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}
CLASSES = ["T-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
# superordinate groups (for clustering beyond single categories)
SUPERORDINATE = {"tops": [0, 2, 4, 6], "footwear": [5, 7, 9], "other": [1, 3, 8]}


def _fetch(name: str) -> Path:
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL + name, path)
    return path


def load_fashion_mnist(split: str = "train", n: int | None = None, size: int = 64, seed: int = 0, raw: bool = False):
    """(images (N, 1, size, size) float, labels (N,) long). ``n`` draws a
    class-balanced random subset. ``raw=True`` returns the native (N, 28, 28)
    items in [0, 1] instead (for composing cluttered scenes)."""
    img_file, lab_file = FILES[split]
    x = np.frombuffer(gzip.open(_fetch(img_file)).read(), np.uint8, offset=16).reshape(-1, 28, 28)
    y = np.frombuffer(gzip.open(_fetch(lab_file)).read(), np.uint8, offset=8)
    x, y = torch.from_numpy(x.copy()).float() / 255.0, torch.from_numpy(y.copy()).long()
    if n is not None:
        g = torch.Generator().manual_seed(seed)
        per = n // len(CLASSES)
        idx = torch.cat([torch.nonzero(y == c).flatten()[torch.randperm(int((y == c).sum()), generator=g)[:per]]
                         for c in range(len(CLASSES))])
        idx = idx[torch.randperm(len(idx), generator=g)]
        x, y = x[idx], y[idx]
    if raw:
        return x, y
    up = F.interpolate(x.unsqueeze(1), scale_factor=2, mode="bilinear", align_corners=False)
    pad = (size - up.shape[-1]) // 2
    canvas = F.pad(up, (pad, size - up.shape[-1] - pad, pad, size - up.shape[-1] - pad))
    return canvas - canvas.mean((2, 3), keepdim=True), y
