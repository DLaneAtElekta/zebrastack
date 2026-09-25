"""Upper-stage Gabors as learnable spatial kernels (Gabor-initialized).

``SpatialGaborBank`` is a drop-in replacement for ``GaborBank`` at V4 / PIT /
AIT: the same quadrature filters, applied per input channel, but as spatial
convolution kernels that can be trained. Each filter is initialized from the
fixed bank, so with learning off the stage computes what ``HigherStage``
computes (up to kernel truncation and border handling); a tether to the
initial weights keeps learned filters "on rails".

``LearnedHigherStage`` is ``HigherStage`` with this bank. Everything else in
the block (energy, gain, normalization, log, pooling, pass-through, whitening,
TICA) is unchanged.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .gabor import GaborBank
from .hierarchy import HigherStage


class SpatialGaborBank(nn.Module):
    """``n_orientations`` quadrature Gabors per input channel as (real, imag)
    spatial kernel pairs of size ``kernel_size``. Output: (B, C * N, H, W)
    complex, ordered (in_channel, orientation), like ``GaborBank``."""

    def __init__(self, in_channels: int, n_orientations: int = 4, freq: float = 0.25, kernel_size: int = 15,
                 bandwidth_octaves: float = 1.2, aspect: float = 0.7):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        ref = GaborBank(n_orientations, 1, freq, bandwidth_octaves, aspect)
        big = 4 * kernel_size + 1  # build large, then crop: avoids the wrap-around of a small FFT grid
        k = ref.kernels(big)  # (N, big, big) complex, centered at big // 2
        c, r = big // 2, kernel_size // 2
        k = k[:, c - r:c + r + 1, c - r:c + r + 1]
        # conv2d is correlation; flip so outputs equal the (convolving) frequency-domain bank
        k = torch.flip(k, (-2, -1))
        pair = torch.stack([k.real, k.imag], 1).float()  # (N, 2, k, k)
        w = pair.unsqueeze(0).expand(in_channels, *pair.shape).reshape(-1, 1, kernel_size, kernel_size)
        self.in_channels, self.n_filters, self.radius = in_channels, n_orientations, r
        self.weight = nn.Parameter(w.clone())
        self.register_buffer("weight_init", w.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        r = self.radius
        ph, pw = min(r, h - 1), min(r, w - 1)
        xp = F.pad(x, (pw, pw, ph, ph), mode="reflect")
        xp = F.pad(xp, (r - pw, r - pw, r - ph, r - ph))  # zeros beyond what reflection can supply
        out = F.conv2d(xp, self.weight, groups=c).view(b, c, self.n_filters, 2, h, w)
        return torch.complex(out[:, :, :, 0], out[:, :, :, 1]).reshape(b, c * self.n_filters, h, w)

    def tether(self) -> torch.Tensor:
        """Squared distance from the Gabor initialization, relative to its norm."""
        return (self.weight - self.weight_init).pow(2).sum() / self.weight_init.pow(2).sum()

    def drift(self) -> float:
        with torch.no_grad():
            return float(self.tether().sqrt())


class LearnedHigherStage(HigherStage):
    """``HigherStage`` whose second-order Gabors are learnable spatial kernels."""

    def __init__(self, name: str, in_channels: int, sheet: int, radius: int = 1, freq: float = 0.25,
                 n_orientations: int = 4, kernel_size: int = 15, **kw):
        super().__init__(name, in_channels, sheet, radius, freq=freq, n_orientations=n_orientations, **kw)
        self.bank = SpatialGaborBank(in_channels, n_orientations, freq, kernel_size)
