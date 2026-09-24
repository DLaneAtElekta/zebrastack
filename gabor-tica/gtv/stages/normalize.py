"""Divisive normalization and log (plan Phase 1, block step [4]).

Heeger-style: each unit's energy is divided by a semi-saturation constant plus
a spatially blurred average of energy across the normalization pool. Under a
full-field stimulus the pool scales with contrast just like the unit, so tuning
becomes contrast invariant once contrast is well above ``sigma``.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def gaussian_blur(x: torch.Tensor, std: float) -> torch.Tensor:
    """Separable Gaussian blur of (B, C, H, W) with reflect padding."""
    if std <= 0:
        return x
    r = math.ceil(3 * std)
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / std) ** 2)
    c = x.shape[1]
    h, w = x.shape[-2:]
    ph, pw = min(r, h - 1), min(r, w - 1)
    x = F.pad(x, (pw, pw, ph, ph), mode="reflect")
    # a kernel longer than the padded input is cropped to the padding actually
    # available, then renormalized so constants are preserved
    kh = k[r - ph : r + ph + 1]
    kw = k[r - pw : r + pw + 1]
    x = F.conv2d(x, (kh / kh.sum()).view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    x = F.conv2d(x, (kw / kw.sum()).view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return x


class DivisiveNormalization(nn.Module):
    """R_i = E_i / (sigma**2 + blur(mean_{j in pool(i)} E_j)).

    ``pool_groups`` splits channels into equal groups that normalize
    separately (e.g. one group per scale); 1 pools over all channels.
    """

    def __init__(self, sigma: float = 0.05, spatial_std: float = 4.0, pool_groups: int = 1):
        super().__init__()
        self.sigma = sigma
        self.spatial_std = spatial_std
        self.pool_groups = pool_groups

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        b, c, h, w = e.shape
        g = self.pool_groups
        pool = e.view(b, g, c // g, h, w).mean(dim=2)
        pool = gaussian_blur(pool, self.spatial_std)
        pool = pool.repeat_interleave(c // g, dim=1)
        return e / (self.sigma**2 + pool)


class Log(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.log(r + self.eps)
