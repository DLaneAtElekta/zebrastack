"""Recognition-path stage blocks (plan section 2.1).

Planned modules:
    gabor.py      Phase 1  quadrature Gabor banks, energy, log-polar sampling
    normalize.py  Phase 1  Heeger-style divisive normalization + log
    tica.py       Phase 2  TICA / RICA loss, topographic neighborhoods
    block.py      Phase 0+ stage-block base class and the full block
"""

from .block import IdentityStage, Stage, build_stage, build_stack

__all__ = ["IdentityStage", "Stage", "build_stage", "build_stack"]
