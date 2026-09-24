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

__all__ = [
    "contrast_modulated_grating",
    "curvature_fragment",
    "drifting_grating",
    "grating",
    "oriented_noise",
    "texture_pair",
]
