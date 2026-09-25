"""Recognition-path stage blocks (plan section 2.1).

Modules:
    block.py      stage-block base class, IdentityStage, V1Stage, registry
    gabor.py      quadrature Gabor bank and energy (log-polar sampling: Phase 5)
    normalize.py  Heeger-style divisive normalization + log
    tica.py       TICA / RICA, whitening, topographic sheet arrangement
    v2.py         V2 stage: Design A (learned) and Design B (second-order Gabor)
    hierarchy.py  V4 / PIT / AIT: repeats of the Design B block (Phase 5)
"""

from .block import IdentityStage, Stage, V1Stage, build_stack, build_stage
from .gabor import GaborBank, energy
from .normalize import DivisiveNormalization, Log
from .tica import TICA, Whitener, arrange_on_sheet, torus_neighborhood
from .v2 import SecondOrderFeatures, V2Stage
from .hierarchy import HigherStage
from .block import STAGE_KINDS

STAGE_KINDS["v2_tica"] = V2Stage  # registered here to avoid a block <-> v2 import cycle

__all__ = [
    "DivisiveNormalization",
    "GaborBank",
    "HigherStage",
    "IdentityStage",
    "Log",
    "SecondOrderFeatures",
    "Stage",
    "TICA",
    "V2Stage",
    "Whitener",
    "arrange_on_sheet",
    "torus_neighborhood",
    "V1Stage",
    "build_stack",
    "build_stage",
    "energy",
]
