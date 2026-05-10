"""Preferences C(o) — soul.md §2.6.

These are not rewards. They are the prior over observations that *defines
the phenotype*. Surprise = -log P(o) under this prior is what the agent
minimises.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pomdp import Observation, PlausibilityState


@dataclass
class Preferences:
    min_neural_alignment: float = 0.2
    min_accuracy: float = 0.6
    min_topography_score: float = 0.5
    min_plausibility: float = 0.7
    max_ece: float = 0.1

    def surprise(self, obs: Observation, bio: PlausibilityState) -> float:
        """Scalar surprise; lower is more in-phenotype.

        Each preferred outcome contributes a hinge penalty for falling
        short. This is intentionally crude — replace with a proper
        log-density once probes are calibrated.
        """
        s = 0.0
        acc = obs.metrics.get("accuracy", 0.0)
        s += max(0.0, self.min_accuracy - acc)

        ece = obs.metrics.get("ece", 1.0)
        s += max(0.0, ece - self.max_ece)

        topo = obs.topography.get("score", 0.0)
        s += max(0.0, self.min_topography_score - topo)

        plaus = sum(bio.as_vector()) / len(bio.as_vector())
        s += max(0.0, self.min_plausibility - plaus)

        align = obs.metrics.get("neural_alignment_mean", 0.0)
        s += max(0.0, self.min_neural_alignment - align)

        return s
