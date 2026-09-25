"""Temporal coherence (Phase 4).

Modules:
    coherence.py   bubbles-style spatiotemporal energy pooling and temporal
                   coherence of activity levels, for TICA on frame pairs
    filters.py     the same objectives training an upper stage's
                   Gabor-initialized filters (LearnedHigherStage)
"""

from .coherence import bubbles_loss, coherence_loss, fit_temporal
from .filters import fit_stage_filters

__all__ = ["bubbles_loss", "coherence_loss", "fit_temporal", "fit_stage_filters"]
