"""Data loaders (plan section 5).

Available: natural-image patches (``natural``; repo photos until van Hateren is
added) and simulated drift sequences (``sequences``, Phase 4). Category data: Fashion-MNIST (``categories``). Planned: video, other category
subsets (Phase 5-7), retinal-wave-like noise (optional Phase 1 variant).
Probes live in ``gtv.probes``.
"""

from .categories import CLASSES, SUPERORDINATE, load_fashion_mnist
from .natural import REPO_PHOTOS, load_grayscale, random_patches
from .sequences import affine, center_crop, drift_pairs

__all__ = [
    "CLASSES",
    "SUPERORDINATE",
    "load_fashion_mnist",
    "REPO_PHOTOS", "affine", "center_crop", "drift_pairs", "load_grayscale", "random_patches"]
