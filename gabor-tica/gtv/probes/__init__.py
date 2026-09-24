"""Synthetic probe stimuli with known ground truth (plan principle 3).

Phase 0 provides gratings, drifting gratings, contrast-modulated (second-order)
gratings, oriented texture pairs and curvature fragments. Cluttered target
scenes arrive with the category data in Phase 5/6.
"""

from .stimuli import (
    contrast_modulated_grating,
    curvature_fragment,
    drifting_grating,
    grating,
    oriented_noise,
    texture_pair,
)
from .tuning import (
    center,
    excess_kurtosis,
    grating_responses,
    half_width_half_height,
    phase_modulation,
    skewness,
)

__all__ = [
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
