"""Recognition-path stage blocks (plan section 2.1).

Modules:
    block.py      stage-block base class, IdentityStage, V1Stage, registry
    gabor.py      quadrature Gabor bank and energy (log-polar sampling: Phase 5)
    normalize.py  Heeger-style divisive normalization + log
    tica.py       Phase 2: TICA / RICA loss, topographic neighborhoods
"""

from .block import IdentityStage, Stage, V1Stage, build_stack, build_stage
from .gabor import GaborBank, energy
from .normalize import DivisiveNormalization, Log

__all__ = [
    "DivisiveNormalization",
    "GaborBank",
    "IdentityStage",
    "Log",
    "Stage",
    "V1Stage",
    "build_stack",
    "build_stage",
    "energy",
]
