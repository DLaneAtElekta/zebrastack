"""Gabor-fit probe for the V1 stage — soul.md §4 constraint 3.

Given a stack of V1-stage filters of shape (H, W, C_out), fit a real-valued
Gabor (gaussian-modulated cosine) to each output channel and report the
normalised residual. Low residuals => the filter bank is Gabor-like, which
is what the phenotype expects.

No TF dependency: numpy in, numpy out. Wires to the analytical Gabor bank
in `fashionmnist/oriented_powermap_2d.py` and to learned filters from any
backend that can hand us a numpy array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares


def _gabor(
    params: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    cx, cy, theta, sigma, freq, phase, amp = params
    xr = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
    yr = -(x - cx) * np.sin(theta) + (y - cy) * np.cos(theta)
    envelope = np.exp(-(xr * xr + yr * yr) / (2.0 * max(sigma, 1e-3) ** 2))
    carrier = np.cos(2.0 * np.pi * freq * xr + phase)
    return amp * envelope * carrier


def _initial_params(filt: np.ndarray) -> np.ndarray:
    h, w = filt.shape
    return np.array(
        [0.0, 0.0, 0.0, 0.25 * min(h, w), 1.0 / max(h, w), 0.0, float(np.max(np.abs(filt)) or 1.0)]
    )


def gabor_fit_residual(filt: np.ndarray) -> float:
    """Fit a Gabor to one filter and return normalised residual in [0, ~1]."""
    h, w = filt.shape
    ys, xs = np.mgrid[-(h // 2) : h - h // 2, -(w // 2) : w - w // 2]
    target = filt.astype(np.float64)
    target_energy = float(np.sum(target * target)) + 1e-12

    def residuals(p: np.ndarray) -> np.ndarray:
        return (_gabor(p, xs.astype(np.float64), ys.astype(np.float64)) - target).ravel()

    try:
        result = least_squares(residuals, _initial_params(target), max_nfev=200)
        rss = float(np.sum(result.fun * result.fun))
        return rss / target_energy
    except Exception:
        return 1.0


@dataclass
class GaborFitProbe:
    """Aggregate probe over a (H, W, C_out) filter bank."""

    name: str = "gabor_fit"

    def __call__(self, filters_hwc: np.ndarray) -> Tuple[float, List[float]]:
        if filters_hwc.ndim != 3:
            raise ValueError(f"expected (H, W, C_out), got {filters_hwc.shape}")
        per_channel = [
            gabor_fit_residual(filters_hwc[..., c])
            for c in range(filters_hwc.shape[-1])
        ]
        score = float(np.exp(-np.mean(per_channel)))
        return score, per_channel
