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
        first_budget: int | str | None = None,
    ):
        """``first_budget``: whitening dimensions for the first-order (+ skip)
        channels, i.e. the pass-through of the stage below. None = half the
        sheet (the Phase 5 default, which discards part of what the stage
        received); "all" = every first-order channel, the rest of the sheet
        going to second-order channels."""
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
        if first_budget == "all":
            d1 = self.n_first
        elif first_budget is None:
            d1 = min(self.n_first, dim // 2)
        else:
            d1 = int(first_budget)
        if not 0 < d1 < dim:
            raise ValueError(f"first-order budget {d1} must leave room on a {dim}-unit sheet")
        self.dim = dim
        whitener = GroupWhitener([(0, self.n_first, d1), (self.n_first, self.n_first + self.n_second, dim - d1)])
        self.tica = TICA(sheet, sheet, radius, eps, whitener)

    def features(self, x: torch.Tensor, skips: list[torch.Tensor] | None = None,
                 gain: torch.Tensor | None = None, noise_T: float | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """(B, C, H, W) [+ skip maps] -> (B, n_first + n_second, H/2, W/2).

        ``gain`` (block step [3], the thalamic attention field): multiplies the
        Gabor energies before divisive normalization, drive = A * E, so
        R = drive / (sigma + pool(drive)) (normalization model of attention).
        Shape (n_second,) for feature-only gain, or broadcastable to
        (B, n_second, H, W) for spatial x feature gain.

        ``noise_T``: a response-noise bottleneck on everything the stage passes
        up (smaller T = noisier). The gained energies get Poisson-like noise on
        the normalized responses, R + sqrt(R / T) * N(0, 1) clamped at 0, before
        the log (so attention's gain raises their signal-to-noise). The
        first-order pass-through channels get Gaussian noise with standard
        deviation (channel spread) / sqrt(T), so information cannot bypass
        the bottleneck. Needs a fitted stage (the spread comes from its whitener)."""
        e = energy(self.bank(x))
        if gain is not None:
            e = e * (gain.view(1, -1, 1, 1) if gain.dim() == 1 else gain)
        r = self.norm(e)
        if noise_T is not None:
            noise = torch.randn(r.shape, generator=generator)
            r = (r + torch.sqrt(r.clamp(min=0) / noise_T) * noise).clamp(min=0)
        second = F.avg_pool2d(self.log(r), 2)
        h, w = second.shape[-2:]
        first = [F.adaptive_avg_pool2d(x, (h, w))]
        for s in skips or []:
            first.append(F.adaptive_avg_pool2d(s, (h, w)))
        first = torch.cat(first, 1)
        if noise_T is not None:
            spread = self.tica.whitener.parts[0].scale.view(1, -1, 1, 1)
            first = first + spread / noise_T**0.5 * torch.randn(first.shape, generator=generator)
        return torch.cat([first, second], 1)

    def fit(self, x: torch.Tensor, skips: list[torch.Tensor] | None = None, border: int = 0,
            chunk: int = 256, **tica_kw) -> list[float]:
        with torch.no_grad():
            f = torch.cat([self.features(xb, [s[i:i + chunk] for s in skips] if skips else None)
                           for i, xb in zip(range(0, len(x), chunk), x.split(chunk))])
        if border:
            f = f[..., border:-border, border:-border]
        vecs = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
        return self.tica.fit(vecs, self.dim, **tica_kw)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor] | None = None,
                gain: torch.Tensor | None = None, noise_T: float | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        f = self.features(x, skips, gain, noise_T, generator)
        m, b = self.tica.affine
        return torch.einsum("nd,bdhw->bnhw", m, f) + b.view(1, -1, 1, 1)

    def pooled_energy(self, s: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.einsum("ij,bjhw->bihw", self.tica.h, s.pow(2)) + self.tica.eps)

    def out_channels(self, in_channels: int) -> int:
        return self.tica.n_units


    def features_from_outputs(self, s: torch.Tensor) -> torch.Tensor:
        """Top-down: the stage's input features implied by outputs ``s``
        (B, n_units, H, W), via the pseudo-inverse of whitening + TICA (the
        minimum-norm features in the kept subspace, plus the feature mean)."""
        m, b = self.tica.affine
        pinv = torch.linalg.pinv(m)  # (D, n)
        return torch.einsum("dn,bnhw->bdhw", pinv, s - b.view(1, -1, 1, 1))

    @property
    def second_order_slice(self) -> slice:
        return slice(self.n_first, self.n_first + self.n_second)


def build_stack(stage_cfgs: dict, in_channels: dict[str, int]) -> dict[str, "HigherStage"]:
    """Instantiate V4/PIT/AIT from a Phase 5 ``stages`` config. ``in_channels``
    maps "V1" and "V2" to their channel counts."""
    stages, prev = {}, in_channels["V2"]
    for name, sc in stage_cfgs.items():
        skip = sc.get("skip")
        stages[name] = HigherStage(name, prev, sc["sheet"], sc["radius"],
                                   skip_channels=in_channels[skip] if skip else 0,
                                   first_budget=sc.get("first_budget"))
        in_channels[name] = stages[name].tica.n_units
        prev = in_channels[name]
    return stages
