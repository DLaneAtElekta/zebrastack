"""Natural-image patches from a folder of photos.

Until a van Hateren set is downloaded, the repo's own photos
(``../Data/DougsCars``) serve as the natural-statistics source.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_PHOTOS = Path(__file__).resolve().parents[3] / "Data" / "DougsCars"


def load_grayscale(folder: str | Path = REPO_PHOTOS, max_side: int = 512, limit: int | None = None) -> list[torch.Tensor]:
    """Load images as log-luminance tensors (H, W), downsampled so the long
    side is at most ``max_side`` (downsampling also averages away JPEG noise)."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    if not paths:
        raise FileNotFoundError(f"no images in {folder}")
    out = []
    for p in paths[:limit]:
        img = Image.open(p).convert("L")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        a = np.asarray(img, dtype=np.float32) / 255.0
        out.append(torch.log(torch.from_numpy(a) + 1e-2))
    return out


def random_patches(
    images: list[torch.Tensor],
    n: int,
    size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(n, 1, size, size) crops, each made zero-mean, then jointly scaled to unit std."""
    patches = []
    for _ in range(n):
        img = images[int(torch.randint(len(images), (1,), generator=generator))]
        h, w = img.shape
        y = int(torch.randint(h - size + 1, (1,), generator=generator))
        x = int(torch.randint(w - size + 1, (1,), generator=generator))
        p = img[y : y + size, x : x + size]
        patches.append(p - p.mean())
    batch = torch.stack(patches).unsqueeze(1)
    return batch / (batch.std() + 1e-8)
