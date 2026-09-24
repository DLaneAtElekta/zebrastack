"""Stage-block interface.

Every ventral area (V1, V2, V4, PIT, AIT) is a ``Stage``: it maps a map of
shape (B, C, H, W) to (B, C', H', W'). Phase 0 ships only ``IdentityStage`` so
the data -> stage -> viz pipeline can be exercised end to end; later phases
register real blocks under new ``kind`` names.
"""

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .gabor import GaborBank, energy
from .normalize import DivisiveNormalization, Log


class Stage(nn.Module):
    """Base class for one ventral area."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def out_channels(self, in_channels: int) -> int:
        raise NotImplementedError


class IdentityStage(Stage):
    """Pass-through stage used to validate the pipeline (Phase 0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def out_channels(self, in_channels: int) -> int:
        return in_channels


class V1Stage(Stage):
    """Developmental V1 (Phase 1): fixed quadrature Gabors -> energy ->
    divisive normalization -> log -> optional average-pool decimation.

    Output channels are complex cells ordered (in_channel, scale, orientation).
    """

    def __init__(
        self,
        name: str = "V1",
        n_orientations: int = 8,
        n_scales: int = 3,
        max_freq: float = 0.25,
        bandwidth_octaves: float = 1.2,
        aspect: float = 0.7,
        norm_sigma: float = 0.05,
        norm_spatial_std: float = 4.0,
        norm_pool_groups: int = 1,
        log_eps: float = 1e-4,
        decimate: int = 1,
    ):
        super().__init__(name)
        self.bank = GaborBank(n_orientations, n_scales, max_freq, bandwidth_octaves, aspect)
        self.norm = DivisiveNormalization(norm_sigma, norm_spatial_std, norm_pool_groups)
        self.log = Log(log_eps)
        self.decimate = decimate

    def parts(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Intermediate maps for diagnostics: energy, normalized, log, output."""
        e = energy(self.bank(x))
        r = self.norm(e)
        lg = self.log(r)
        out = F.avg_pool2d(lg, self.decimate) if self.decimate > 1 else lg
        return {"energy": e, "normalized": r, "log": lg, "output": out}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.parts(x)["output"]

    def out_channels(self, in_channels: int) -> int:
        return in_channels * self.bank.n_filters


STAGE_KINDS: dict[str, Callable[..., Stage]] = {
    "identity": IdentityStage,
    "v1_gabor": V1Stage,
}


def build_stage(cfg: dict) -> Stage:
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    if kind not in STAGE_KINDS:
        raise KeyError(f"unknown stage kind {kind!r}; known: {sorted(STAGE_KINDS)}")
    return STAGE_KINDS[kind](**cfg)


def build_stack(stage_cfgs: list[dict]) -> nn.Sequential:
    return nn.Sequential(*(build_stage(c) for c in stage_cfgs))
