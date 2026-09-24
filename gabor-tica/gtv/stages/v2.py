"""V2 stage (plan Phase 2): two designs feeding the same TICA reduction.

Design B (primary) - fixed second-order Gabors on each V1 channel, i.e. a
    filter-rectify-filter / scattering path: V1 log-energy map -> coarser
    quadrature Gabors -> energy -> divisive normalization -> log -> 2x pool.
    Only frequency-decreasing paths are kept (second-order frequency at most
    ``path_ratio`` times the parent V1 frequency). The pooled first-order V1
    maps are appended, as in a scattering transform. TICA then runs across
    channels at each location (a 1x1 reduction). ``whiten_groups`` gives the
    first- and second-order groups separate PCA budgets; with one shared PCA
    the second-order channels crowd out the first-order ones.

Design A (baseline) - no fixed second stage: TICA learns filters directly on
    ``patch`` x ``patch`` neighborhoods of all V1 channels (a strided conv).

Both output TICA coefficients s of shape (B, n_units, H/2, W/2) where H, W is
the V1 map size. Frequencies are given in cycles per *input-image* pixel.
"""

import torch
import torch.nn.functional as F

from .block import Stage
from .gabor import GaborBank, energy
from .normalize import DivisiveNormalization, Log
from .tica import TICA, GroupWhitener


class SecondOrderFeatures(torch.nn.Module):
    """Design B front end: V1 maps (B, C1, H, W) -> (B, C1_pooled + C2, H/2, W/2)."""

    def __init__(
        self,
        v1_freqs: list[float],
        v1_decimate: int,
        freqs: tuple[float, ...] = (0.0625, 0.03125),
        n_orientations: int = 4,
        bandwidth_octaves: float = 1.2,
        aspect: float = 0.7,
        path_ratio: float = 0.5,
        norm_sigma: float = 0.1,
        norm_spatial_std: float = 2.0,
        log_eps: float = 1e-4,
        include_first_order: bool = True,
    ):
        super().__init__()
        self.freqs = sorted(freqs, reverse=True)
        self.v1_decimate = v1_decimate
        # one bank per second-order scale, in V1-map pixel units
        self.banks = torch.nn.ModuleList(
            GaborBank(n_orientations, 1, f * v1_decimate, bandwidth_octaves, aspect) for f in self.freqs
        )
        self.n_orientations = n_orientations
        # keep[j2] = V1 channels whose frequency is at least f2 / path_ratio
        self.keep = [
            [c for c, f1 in enumerate(v1_freqs) if f2 <= path_ratio * f1 + 1e-12] for f2 in self.freqs
        ]
        self.n_first = len(v1_freqs) if include_first_order else 0
        self.n_second = sum(len(k) for k in self.keep) * n_orientations
        self.norm = DivisiveNormalization(norm_sigma, norm_spatial_std)
        self.log = Log(log_eps)
        self.include_first_order = include_first_order

    @property
    def n_channels(self) -> int:
        return self.n_first + self.n_second

    def forward(self, v1: torch.Tensor) -> torch.Tensor:
        second = []
        for bank, keep in zip(self.banks, self.keep):
            if keep:
                second.append(energy(bank(v1[:, keep])))
        e2 = torch.cat(second, 1)
        out = F.avg_pool2d(self.log(self.norm(e2)), 2)
        if self.include_first_order:
            out = torch.cat([F.avg_pool2d(v1, 2), out], 1)
        return out


class V2Stage(Stage):
    """V2 = (Design A | Design B front end) + TICA on a ``sheet`` x ``sheet`` torus.

    Call ``fit(v1_maps)`` once on natural V1 outputs before using ``forward``.
    """

    def __init__(
        self,
        name: str = "V2",
        design: str = "B",
        v1_freqs: list[float] | None = None,
        v1_decimate: int = 2,
        sheet: int = 8,
        radius: int = 1,
        dim: int | None = None,
        tica_mode: str = "complete",
        patch: int = 8,
        second_order: dict | None = None,
        eps: float = 1e-3,
        whiten_groups: list[int] | None = None,
    ):
        super().__init__(name)
        if design not in ("A", "B"):
            raise ValueError("design must be 'A' or 'B'")
        self.design = design
        self.patch = patch
        self.tica_mode = tica_mode
        self.dim = dim or sheet * sheet
        self.front = None
        whitener = None
        if design == "B":
            if v1_freqs is None:
                raise ValueError("design B needs v1_freqs (one per V1 channel)")
            self.front = SecondOrderFeatures(v1_freqs, v1_decimate, **(second_order or {}))
            if whiten_groups is not None:
                # [first-order dims, second-order dims]: separate PCA budgets so
                # the 160 second-order channels cannot crowd out the first-order ones
                d1, d2 = whiten_groups
                if d1 + d2 != self.dim or not self.front.include_first_order:
                    raise ValueError("whiten_groups needs first-order channels and must sum to dim")
                nf, nc = self.front.n_first, self.front.n_channels
                whitener = GroupWhitener([(0, nf, d1), (nf, nc, d2)])
        self.tica = TICA(sheet, sheet, radius, eps, whitener)

    # ---- sampling training vectors
    def _vectors(self, v1: torch.Tensor, n_per_image: int, border: int, gen: torch.Generator) -> torch.Tensor:
        if self.design == "B":
            f = self.front(v1)  # (B, C, h, w)
            f = f[:, :, border : f.shape[2] - border, border : f.shape[3] - border]
            return f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
        # design A: random patch x patch neighborhoods of the V1 maps
        b, c, h, w = v1.shape
        p = self.patch
        ys = torch.randint(0, h - p + 1, (b, n_per_image), generator=gen)
        xs = torch.randint(0, w - p + 1, (b, n_per_image), generator=gen)
        out = [v1[i, :, y : y + p, x : x + p].reshape(-1) for i in range(b) for y, x in zip(ys[i].tolist(), xs[i].tolist())]
        return torch.stack(out)

    def fit(self, v1: torch.Tensor, n_per_image: int = 64, border: int = 2, seed: int = 0, **tica_kw) -> list[float]:
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            x = self._vectors(v1, n_per_image, border, gen)
        return self.tica.fit(x, self.dim, mode=self.tica_mode, seed=seed, **tica_kw)

    def training_vectors(self, v1: torch.Tensor, n_per_image: int = 64, border: int = 2, seed: int = 0) -> torch.Tensor:
        return self._vectors(v1, n_per_image, border, torch.Generator().manual_seed(seed))

    # ---- inference
    def features(self, v1: torch.Tensor) -> torch.Tensor:
        """Pre-TICA representation (design B only): (B, C, H/2, W/2)."""
        if self.front is None:
            raise AttributeError("design A has no fixed pre-TICA features")
        return self.front(v1)

    def forward(self, v1: torch.Tensor) -> torch.Tensor:
        m, bias = self.tica.affine  # (n, D), (n,)
        n = m.shape[0]
        if self.design == "B":
            f = self.front(v1)
            return torch.einsum("nd,bdhw->bnhw", m, f) + bias.view(1, n, 1, 1)
        c = v1.shape[1]
        p = self.patch
        kernel = m.view(n, c, p, p)
        # stride 2 halves the map; padding keeps the output H/2 x W/2
        pad = (p - 2) // 2
        return F.conv2d(v1, kernel, bias, stride=2, padding=pad)

    def pooled_energy(self, s: torch.Tensor) -> torch.Tensor:
        """Complex-cell-like TICA output: sqrt of sheet-neighborhood energy,
        (B, n, H, W) -> (B, n, H, W)."""
        return torch.sqrt(torch.einsum("ij,bjhw->bihw", self.tica.h, s.pow(2)) + self.tica.eps)

    def out_channels(self, in_channels: int) -> int:
        return self.tica.n_units
