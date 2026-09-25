"""Recognition-path stage blocks (plan section 2.1).

Modules:
    block.py      stage-block base class, IdentityStage, V1Stage, registry
    gabor.py      quadrature Gabor bank and energy (log-polar sampling: Phase 5)
    normalize.py  Heeger-style divisive normalization + log
    tica.py       TICA / RICA, whitening, topographic sheet arrangement
    v2.py         V2 stage: Design A (learned) and Design B (second-order Gabor)
    hierarchy.py  V4 / PIT / AIT: repeats of the Design B block (Phase 5)
    learned_gabor.py  the same block with Gabor-initialized learnable kernels
"""

from .block import IdentityStage, Stage, V1Stage, build_stack, build_stage
from .gabor import GaborBank, energy
from .normalize import DivisiveNormalization, Log
from .tica import TICA, Whitener, arrange_on_sheet, torus_neighborhood
from .v2 import SecondOrderFeatures, V2Stage
from .hierarchy import HigherStage, build_stack as build_higher_stack
from .learned_gabor import LearnedHigherStage, SpatialGaborBank
from .block import STAGE_KINDS

# registered here to avoid block <-> v2 / hierarchy import cycles
STAGE_KINDS["v2_tica"] = V2Stage
STAGE_KINDS["higher_gabor"] = HigherStage
STAGE_KINDS["higher_learned"] = LearnedHigherStage

__all__ = [
    "DivisiveNormalization",
    "GaborBank",
    "HigherStage",
    "LearnedHigherStage",
    "SpatialGaborBank",
    "build_higher_stack",
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
