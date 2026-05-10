"""POMDP state and observation types — soul.md §2.1, §2.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

LaminarTag = Literal["FF", "FB", "lateral"]
Stream = Literal["ventral", "dorsal", "temporal"]


@dataclass(frozen=True)
class AreaNode:
    """A cortical-area analogue node in the Van Essen graph."""

    name: str
    stream: Stream
    layout_shape: Tuple[int, int]
    feature_kind: str
    rf_size_px: float
    notes: str = ""


@dataclass(frozen=True)
class AreaEdge:
    """A directed connection between two area nodes, laminar-tagged."""

    src: str
    dst: str
    laminar: LaminarTag
    weight_init: str = "default"


@dataclass
class ArchitectureState:
    """Van Essen-style distributed graph; not a Sequential stack."""

    nodes: List[AreaNode] = field(default_factory=list)
    edges: List[AreaEdge] = field(default_factory=list)
    front_end: str = "anisotropic_diffusion+CLAHE"
    readout: str = "topographic_latent_sheet"

    def levels(self) -> Dict[str, int]:
        """Felleman–Van Essen partial-order level inferred from FF edges."""
        level: Dict[str, int] = {n.name: 0 for n in self.nodes}
        changed = True
        guard = 0
        while changed and guard < 64:
            changed = False
            for e in self.edges:
                if e.laminar == "FF" and level[e.dst] <= level[e.src]:
                    level[e.dst] = level[e.src] + 1
                    changed = True
            guard += 1
        return level


@dataclass
class ParameterState:
    """Opaque handle to weights; the agent reasons about provenance, not values."""

    backend: str = "tf"
    checkpoint_uri: Optional[str] = None
    n_params: int = 0
    seed: int = 0


@dataclass
class DataState:
    """Which dataset / split / augmentation regime is shaping the model."""

    dataset: str = "fashion_mnist"
    split: str = "train"
    augmentation: str = "default"


@dataclass
class PlausibilityState:
    """Vector of biological constraints (§4). Each in [0, 1]; 1 = satisfied."""

    locality: float = 0.0
    rf_hierarchy: float = 0.0
    oriented_front_end: float = 0.0
    nonneg_rate_coding: float = 0.0
    divisive_normalisation: float = 0.0
    sparse_coding: float = 0.0
    topographic_organisation: float = 0.0
    distributed_hierarchy: float = 0.0
    plausible_learning: float = 0.0
    energy_budget: float = 0.0
    behavioural_signatures: float = 0.0

    def as_vector(self) -> List[float]:
        return [
            self.locality,
            self.rf_hierarchy,
            self.oriented_front_end,
            self.nonneg_rate_coding,
            self.divisive_normalisation,
            self.sparse_coding,
            self.topographic_organisation,
            self.distributed_hierarchy,
            self.plausible_learning,
            self.energy_budget,
            self.behavioural_signatures,
        ]


@dataclass
class CapabilityState:
    accuracy: float = 0.0
    calibration_ece: float = 1.0
    ood_accuracy: float = 0.0
    shape_bias: float = 0.0
    latent_intrinsic_dim: float = 0.0
    neural_alignment: Dict[str, float] = field(default_factory=dict)


@dataclass
class HiddenState:
    arch: ArchitectureState
    theta: ParameterState
    data: DataState
    bio: PlausibilityState
    cap: CapabilityState


@dataclass
class Observation:
    """What the agent sees each cycle (§2.2)."""

    metrics: Dict[str, float] = field(default_factory=dict)
    probes: Dict[str, Any] = field(default_factory=dict)
    topography: Dict[str, Any] = field(default_factory=dict)
    hierarchy: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
