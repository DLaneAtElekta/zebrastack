"""Temporal coherence (Phase 4).

Modules:
    coherence.py   bubbles-style spatiotemporal energy pooling and temporal
                   coherence of activity levels, for TICA on frame pairs
"""

from .coherence import bubbles_loss, coherence_loss, fit_temporal

__all__ = ["bubbles_loss", "coherence_loss", "fit_temporal"]
