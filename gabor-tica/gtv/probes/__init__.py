"""Synthetic probe stimuli with known ground truth (plan principle 3).

Phase 0 provides gratings, drifting gratings, contrast-modulated (second-order)
gratings, oriented texture pairs, curvature fragments and junctions
(line / L / T / X). Cluttered target
scenes arrive with the category data in Phase 5/6.
"""

from .stimuli import (
    JUNCTION_ARMS,
    junction,
    contrast_modulated_grating,
    curvature_fragment,
    drifting_grating,
    grating,
    oriented_noise,
    texture_pair,
)
from .decode import auc, linear_decode, linear_detect_auc
from .tuning import (
    center,
    excess_kurtosis,
    grating_responses,
    half_width_half_height,
    phase_modulation,
    skewness,
)

__all__ = [
    "auc",
    "linear_decode",
    "linear_detect_auc",
    "JUNCTION_ARMS",
    "junction",
    "center",
    "excess_kurtosis",
    "grating_responses",
    "half_width_half_height",
    "phase_modulation",
    "skewness",
    "contrast_modulated_grating",
    "curvature_fragment",
    "drifting_grating",
    "grating",
    "oriented_noise",
    "texture_pair",
]
