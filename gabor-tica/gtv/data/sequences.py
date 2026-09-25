"""Simulated drift sequences over still images (plan Phase 4 / section 5:
"simulated saccade and drift sequences over still images").

Each pair is two frames of the same scene, the second shifted, rotated and
rescaled slightly, as in fixational drift. Frames are cut from a larger crop
so the transformed frame never sees outside the image.
"""

import math

import torch
import torch.nn.functional as F


def affine(x: torch.Tensor, shift: torch.Tensor, rot_deg: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply per-image affine maps to (B, 1, H, W): content moves by ``shift``
    (B, 2) pixels (x right, y down), rotates by ``rot_deg`` (B,) and is scaled
    by ``scale`` (B,) (> 1 zooms in). Bilinear, reflect padding."""
    b, _, h, w = x.shape
    th = rot_deg * math.pi / 180
    cos, sin = torch.cos(th) / scale, torch.sin(th) / scale
    mat = torch.zeros(b, 2, 3)
    mat[:, 0, 0], mat[:, 0, 1] = cos, -sin
    mat[:, 1, 0], mat[:, 1, 1] = sin, cos
    # affine_grid samples the input at the mapped coordinates, so content moves by -offset
    mat[:, 0, 2] = -2 * shift[:, 0] / w
    mat[:, 1, 2] = -2 * shift[:, 1] / h
    grid = F.affine_grid(mat, (b, 1, h, w), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return x[..., y0 : y0 + size, x0 : x0 + size]


def drift_pairs(
    images: list[torch.Tensor],
    n: int,
    size: int,
    max_shift: float = 2.0,
    max_rot_deg: float = 3.0,
    max_log_scale: float = 0.03,
    margin: int = 16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(n, 2, 1, size, size) frame pairs: frame 1 is frame 0 moved by a random
    shift (uniform within +-max_shift px per axis), rotation and scale change.
    Each crop is standardized with the statistics of its first frame."""
    from .natural import random_patches

    big = random_patches(images, n, size + 2 * margin, generator)  # (n, 1, S, S), jointly scaled
    g = generator
    shift = (torch.rand(n, 2, generator=g) * 2 - 1) * max_shift
    rot = (torch.rand(n, generator=g) * 2 - 1) * max_rot_deg
    scale = torch.exp((torch.rand(n, generator=g) * 2 - 1) * max_log_scale)
    moved = affine(big, shift, rot, scale)
    f0, f1 = center_crop(big, size), center_crop(moved, size)
    mu = f0.mean((2, 3), keepdim=True)
    return torch.stack([f0 - mu, f1 - mu], 1)
