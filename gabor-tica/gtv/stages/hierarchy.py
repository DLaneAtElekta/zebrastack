"""Higher ventral stages (plan Phase 5): V4, PIT, AIT as repeats of the
Design B stage block.

Each ``HigherStage`` takes the stage below's TICA output map (B, C, H, W):
  [1] fixed quadrature Gabors on every input channel (coarse, in map units)
  [2] energy, [4] divisive normalization + log, [5] 2x average pooling
  plus the pooled input channels themselves (first-order, as in a scattering
  transform) and optional skip maps from lower stages, pooled to this size;
  [6] group-whitened TICA across channels on a ``sheet`` x ``sheet`` torus.
"""

import torch
import torch.nn.functional as F

from .block import Stage
from .gabor import GaborBank, energy
from .normalize import DivisiveNormalization, Log
from .tica import TICA, GroupWhitener


class HigherStage(Stage):
    def __init__(
        self,
        name: str,
        in_channels: int,
        sheet: int,
        radius: int = 1,
        freq: float = 0.25,
        n_orientations: int = 4,
        skip_channels: int = 0,
        norm_sigma: float = 0.1,
        norm_spatial_std: float = 1.0,
        log_eps: float = 1e-4,
        eps: float = 1e-3,
    ):
        super().__init__(name)
        self.in_channels, self.skip_channels = in_channels, skip_channels
        self.bank = GaborBank(n_orientations, 1, freq)
        self.norm = DivisiveNormalization(norm_sigma, norm_spatial_std)
        self.log = Log(log_eps)
        self.n_second = in_channels * n_orientations
        self.n_first = in_channels + skip_channels
        dim = sheet * sheet
        if dim > self.n_first + self.n_second:
            raise ValueError("sheet too large for the number of feature channels")
        # split the whitening budget between first-order (+ skip) and second-order channels
        d1 = min(self.n_first, dim // 2)
        self.dim = dim
        whitener = GroupWhitener([(0, self.n_first, d1), (self.n_first, self.n_first + self.n_second, dim - d1)])
        self.tica = TICA(sheet, sheet, radius, eps, whitener)

    def features(self, x: torch.Tensor, skips: list[torch.Tensor] | None = None) -> torch.Tensor:
        """(B, C, H, W) [+ skip maps] -> (B, n_first + n_second, H/2, W/2)."""
        second = F.avg_pool2d(self.log(self.norm(energy(self.bank(x)))), 2)
        h, w = second.shape[-2:]
        first = [F.adaptive_avg_pool2d(x, (h, w))]
        for s in skips or []:
            first.append(F.adaptive_avg_pool2d(s, (h, w)))
        return torch.cat(first + [second], 1)

    def fit(self, x: torch.Tensor, skips: list[torch.Tensor] | None = None, border: int = 0,
            chunk: int = 256, **tica_kw) -> list[float]:
        with torch.no_grad():
            f = torch.cat([self.features(xb, [s[i:i + chunk] for s in skips] if skips else None)
                           for i, xb in zip(range(0, len(x), chunk), x.split(chunk))])
        if border:
            f = f[..., border:-border, border:-border]
        vecs = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
        return self.tica.fit(vecs, self.dim, **tica_kw)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor] | None = None) -> torch.Tensor:
        f = self.features(x, skips)
        m, b = self.tica.affine
        return torch.einsum("nd,bdhw->bnhw", m, f) + b.view(1, -1, 1, 1)

    def pooled_energy(self, s: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.einsum("ij,bjhw->bihw", self.tica.h, s.pow(2)) + self.tica.eps)

    def out_channels(self, in_channels: int) -> int:
        return self.tica.n_units
