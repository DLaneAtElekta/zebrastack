"""Data loaders (plan section 5).

Available: natural-image patches (``natural``; repo photos until van Hateren is
added). Planned: video or simulated saccade sequences (Phase 4), category
subsets (Phase 5-7), retinal-wave-like noise (optional Phase 1 variant).
Probes live in ``gtv.probes``.
"""

from .natural import REPO_PHOTOS, load_grayscale, random_patches

__all__ = ["REPO_PHOTOS", "load_grayscale", "random_patches"]
