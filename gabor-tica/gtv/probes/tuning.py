"""Tuning-curve and distribution measurements for a stage, read at the image
center so boundary handling does not matter."""

import math
from typing import Callable

import torch

from .stimuli import grating

StageFn = Callable[[torch.Tensor], torch.Tensor]  # (B,1,H,W) -> (B,C,H,W)


def center(y: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C) at the central pixel."""
    return y[:, :, y.shape[2] // 2, y.shape[3] // 2]


def grating_responses(
    fn: StageFn,
    size: int,
    freqs: torch.Tensor,
    thetas: torch.Tensor,
    contrast: float = 1.0,
    n_phases: int = 4,
) -> torch.Tensor:
    """Center responses to gratings, averaged over phase: (len(freqs), len(thetas), C)."""
    out = []
    for f in freqs.tolist():
        row = []
        for t in thetas.tolist():
            x = torch.stack(
                [grating(size, f, t, 2 * math.pi * k / n_phases, contrast) for k in range(n_phases)]
            ).unsqueeze(1)
            row.append(center(fn(x)).mean(0))
        out.append(torch.stack(row))
    return torch.stack(out)


def phase_modulation(fn: StageFn, size: int, freq: float, theta: float, n_phases: int = 8) -> torch.Tensor:
    """Per-channel (max - min) / mean of the center response across grating
    phase; ~0 for an ideal complex cell."""
    x = torch.stack([grating(size, freq, theta, 2 * math.pi * k / n_phases) for k in range(n_phases)]).unsqueeze(1)
    r = center(fn(x))
    return (r.max(0).values - r.min(0).values) / (r.mean(0).abs() + 1e-8)


def half_width_half_height(xs: torch.Tensor, curve: torch.Tensor, circular_period: float | None = None) -> float:
    """HWHH of a single-peaked curve (linear interpolation between samples),
    measured on the side of the peak with a crossing, in units of ``xs``."""
    i = int(curve.argmax())
    half = 0.5 * (curve.max() + curve.min())
    n = len(xs)
    widths = []
    for step in (1, -1):
        j = i
        for _ in range(n - 1):
            k = j + step
            if circular_period is None and not 0 <= k < n:
                break
            k %= n
            if curve[k] < half:
                frac = float((curve[j] - half) / (curve[j] - curve[k]))
                dx = float(xs[k] - xs[j])
                if circular_period is not None:
                    dx = (dx + circular_period / 2) % circular_period - circular_period / 2
                d = float(xs[j] - xs[i])
                if circular_period is not None:
                    d = (d + circular_period / 2) % circular_period - circular_period / 2
                widths.append(abs(d + frac * dx))
                break
            j = k
    return min(widths) if widths else float("nan")


def skewness(x: torch.Tensor) -> float:
    x = x.flatten().double()
    z = (x - x.mean()) / x.std()
    return float((z**3).mean())


def excess_kurtosis(x: torch.Tensor) -> float:
    x = x.flatten().double()
    z = (x - x.mean()) / x.std()
    return float((z**4).mean() - 3)
