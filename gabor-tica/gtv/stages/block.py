"""Stage-block interface.

Every ventral area (V1, V2, V4, PIT, AIT) is a ``Stage``: it maps a map of
shape (B, C, H, W) to (B, C', H', W'). Phase 0 ships only ``IdentityStage`` so
the data -> stage -> viz pipeline can be exercised end to end; later phases
register real blocks under new ``kind`` names.
"""

from typing import Callable

import torch
from torch import nn


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


STAGE_KINDS: dict[str, Callable[..., Stage]] = {
    "identity": IdentityStage,
}


def build_stage(cfg: dict) -> Stage:
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    if kind not in STAGE_KINDS:
        raise KeyError(f"unknown stage kind {kind!r}; known: {sorted(STAGE_KINDS)}")
    return STAGE_KINDS[kind](**cfg)


def build_stack(stage_cfgs: list[dict]) -> nn.Sequential:
    return nn.Sequential(*(build_stage(c) for c in stage_cfgs))
