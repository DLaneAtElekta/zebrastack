"""Plausibility and capability probes — one per soul.md §4 constraint.

This stub ships only the V1 Gabor-fit probe; the others are placeholders
so the operating loop has a complete interface to call.
"""

from .gabor_fit import GaborFitProbe, gabor_fit_residual

__all__ = ["GaborFitProbe", "gabor_fit_residual"]
