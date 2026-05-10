"""Phenotyping Agent — see soul.md at repo root.

Self-evidencing active-inference agent that explores a morphospace of
biologically-plausible vision models. This package is the skeleton; the
loop is real but most action handlers are stubs to be filled in.
"""

from .pomdp import (
    ArchitectureState,
    ParameterState,
    DataState,
    PlausibilityState,
    CapabilityState,
    HiddenState,
    Observation,
)
from .actions import Action, ActionKind
from .catalogue import Catalogue, Specimen
from .loop import PhenotypingAgent

__all__ = [
    "ArchitectureState",
    "ParameterState",
    "DataState",
    "PlausibilityState",
    "CapabilityState",
    "HiddenState",
    "Observation",
    "Action",
    "ActionKind",
    "Catalogue",
    "Specimen",
    "PhenotypingAgent",
]
