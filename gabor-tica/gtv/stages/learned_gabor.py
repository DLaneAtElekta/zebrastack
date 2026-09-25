"""Upper-stage Gabors as learnable spatial kernels (Gabor-initialized).

``SpatialGaborBank`` is a drop-in replacement for ``GaborBank`` at V4 / PIT /
AIT: the same quadrature filters, applied per input channel, but as spatial
convolution kernels that can be trained. Each filter is initialized from the
fixed bank, so with learning off the stage computes what ``HigherStage``
computes (up to kernel truncation and border handling); a tether to the
initial weights keeps learned filters "on rails".

``GaborMixBank`` (Phase 5d) keeps the fixed Gabors as a smooth basis and
learns how to combine them: each output filter is a complex-weighted sum of
the basis responses of the input channels near its own channel on the stage
below's TICA sheet, at a few orientations and spatial offsets. Initialized
to the identity (each output = its own channel's Gabor), it computes exactly
what ``GaborBank`` does; learning can then build cross-feature combinations
(conjunctions, curvature-like orientation changes across positions) without
pixel-level kernel noise.

``LearnedHigherStage`` is ``HigherStage`` with either bank. Everything else in
the block (energy, gain, normalization, log, pooling, pass-through, whitening,
TICA) is unchanged.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .gabor import GaborBank
from .hierarchy import HigherStage
from .tica import torus_neighborhood


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

    def step_scale(self) -> float:
        """Typical weight size (the trainer scales its learning rate by it)."""
        return float(self.weight_init.pow(2).mean().sqrt())


class GaborMixBank(nn.Module):
    """Learned combinations of fixed Gabor responses (a smooth filter basis).

    Basis: the ``GaborBank`` responses of every input channel (N orientations),
    each also shifted by the spatial ``offsets`` (P positions). Output filter
    (c, o), for input channel c and orientation o, is

        y[c, o] = sum_{j in nbr(c), o', p} A[c, o, j, o', p] * shift_p(r[j, o'])

    with complex A and nbr(c) the channels within ``mix_radius`` of c on the
    input's ``in_rows`` x ``in_cols`` torus sheet. A complex-weighted sum of
    analytic (positive-frequency) responses is still analytic, so energy
    |y|^2 stays phase-invariant. A starts at the identity (A[c, o, c, o, 0] = 1),
    where the bank equals ``GaborBank``. Output (B, C * N, H, W) complex,
    ordered (in_channel, orientation).
    """

    def __init__(self, in_channels: int, n_orientations: int = 4, freq: float = 0.25, mix_radius: int = 1,
                 offset: int = 2, in_rows: int | None = None, bandwidth_octaves: float = 1.2, aspect: float = 0.7):
        super().__init__()
        rows = in_rows or round(in_channels**0.5)
        if in_channels % rows:
            raise ValueError(f"{in_channels} input channels do not fill a sheet with {rows} rows")
        self.base = GaborBank(n_orientations, 1, freq, bandwidth_octaves, aspect)
        self.in_channels, self.n_filters = in_channels, n_orientations
        self.offsets = [(0, 0)] + ([(0, offset), (0, -offset), (offset, 0), (-offset, 0)] if offset else [])
        nb = torus_neighborhood(rows, in_channels // rows, mix_radius)
        # neighbors of each channel, itself first
        nbr = torch.stack([torch.cat([torch.tensor([c]), torch.nonzero(nb[c]).flatten()[
            torch.nonzero(nb[c]).flatten() != c]]) for c in range(in_channels)])
        self.register_buffer("nbr", nbr)  # (C, M)
        c, n, m, p = in_channels, n_orientations, nbr.shape[1], len(self.offsets)
        a = torch.zeros(c, n, m, n, p, 2)
        a[:, torch.arange(n), 0, torch.arange(n), 0, 0] = 1.0
        self.weight = nn.Parameter(a.clone())
        self.register_buffer("weight_init", a.clone())
        # column of each coefficient in the dense (C*N) x (C*N*P) mixing matrix
        j = nbr.view(c, 1, m, 1, 1).expand(c, n, m, n, p)
        o2 = torch.arange(n).view(1, 1, 1, n, 1).expand(c, n, m, n, p)
        pp = torch.arange(p).view(1, 1, 1, 1, p).expand(c, n, m, n, p)
        row = (torch.arange(c).view(c, 1, 1, 1, 1) * n + torch.arange(n).view(1, n, 1, 1, 1)).expand(c, n, m, n, p)
        self.register_buffer("rows_idx", row.reshape(-1))
        self.register_buffer("cols_idx", ((j * n + o2) * p + pp).reshape(-1))

    def mixing_matrix(self) -> torch.Tensor:
        w = torch.complex(self.weight[..., 0], self.weight[..., 1]).reshape(-1)
        c, n, p = self.in_channels, self.n_filters, len(self.offsets)
        mat = torch.zeros(c * n, c * n * p, dtype=w.dtype, device=w.device)
        return mat.index_put((self.rows_idx, self.cols_idx), w, accumulate=True)

    def _shifted(self, r: torch.Tensor) -> torch.Tensor:
        """(B, K, H, W) -> (B, K, P, H, W): responses at each offset (zeros enter at the border)."""
        h, w = r.shape[-2:]
        out = []
        for dy, dx in self.offsets:
            if dy == 0 and dx == 0:
                out.append(r)
                continue
            pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
            rp = torch.complex(F.pad(r.real, pad), F.pad(r.imag, pad))
            y0, x0 = max(-dy, 0), max(-dx, 0)
            out.append(rp[..., y0:y0 + h, x0:x0 + w])
        return torch.stack(out, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        r = self.base(x)  # (B, C * N, H, W)
        basis = self._shifted(r).reshape(b, -1, h, w)  # (B, C * N * P, H, W), ordered (j, o', p)
        return torch.einsum("oi,bihw->bohw", self.mixing_matrix(), basis)

    def tether(self) -> torch.Tensor:
        """Squared distance from the identity mixing, relative to its norm."""
        return (self.weight - self.weight_init).pow(2).sum() / self.weight_init.pow(2).sum()

    def drift(self) -> float:
        with torch.no_grad():
            return float(self.tether().sqrt())

    def step_scale(self) -> float:
        return 1.0

    def offdiagonal_share(self) -> dict[str, float]:
        """Fraction of the learned change (A - I) that falls on other channels,
        other orientations and shifted positions."""
        with torch.no_grad():
            d = (self.weight - self.weight_init).pow(2).sum(-1)  # (C, N, M, N, P)
            tot = float(d.sum()) + 1e-12
            return {"other_channels": float(d[:, :, 1:].sum()) / tot,
                    "other_orientations": float(d.sum() - d.diagonal(dim1=1, dim2=3).sum()) / tot,
                    "shifted_positions": float(d[..., 1:].sum()) / tot}


class LearnedHigherStage(HigherStage):
    """``HigherStage`` whose second-order Gabors are learnable: ``bank="spatial"``
    (free spatial kernels, Phase 5c) or ``bank="mix"`` (learned combinations of
    the fixed Gabors across sheet-neighbor channels, orientations and offsets)."""

    def __init__(self, name: str, in_channels: int, sheet: int, radius: int = 1, freq: float = 0.25,
                 n_orientations: int = 4, kernel_size: int = 15, bank: str = "spatial", mix_radius: int = 1,
                 offset: int = 2, **kw):
        super().__init__(name, in_channels, sheet, radius, freq=freq, n_orientations=n_orientations, **kw)
        if bank == "spatial":
            self.bank = SpatialGaborBank(in_channels, n_orientations, freq, kernel_size)
        elif bank == "mix":
            self.bank = GaborMixBank(in_channels, n_orientations, freq, mix_radius, offset)
        else:
            raise ValueError(f"unknown bank {bank!r}")
