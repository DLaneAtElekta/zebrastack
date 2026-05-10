"""Operating loop — soul.md §8.

    while alive:
        o_t   = observe()
        q(s)  = update_beliefs(o_t)
        G(pi) = expected_free_energy_per_policy()
        a_t   = argmin_pi G(pi)
        s_t+1 = enact(a_t)
        if accepted: catalogue.append(...)

The loop is real. Most enactors are stubs to be filled in as the agent
grows; the structure is what matters here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from typing import Callable, Iterable, List, Optional

from .actions import Action, ActionKind
from .catalogue import Catalogue, Specimen
from .pomdp import HiddenState, Observation
from .preferences import Preferences

log = logging.getLogger(__name__)

Enactor = Callable[[HiddenState, Action], HiddenState]
Observer = Callable[[HiddenState], Observation]
PolicyEnumerator = Callable[[HiddenState], Iterable[Action]]


def default_observer(state: HiddenState) -> Observation:
    """Stub: read whatever is already on the state, no real probes yet."""
    return Observation(
        metrics={"accuracy": state.cap.accuracy, "ece": state.cap.calibration_ece},
        topography={"score": state.bio.topographic_organisation},
    )


def default_policies(state: HiddenState) -> List[Action]:
    """Tiny default policy set — replace with morphospace search."""
    return [
        Action(ActionKind.TRAIN),
        Action(ActionKind.INVESTIGATE, {"probe": "gabor_fit"}),
        Action(ActionKind.PROPOSE, {"mutation": "add_feedback_edge"}),
        Action(ActionKind.REFRAIN),
    ]


def default_enactor(state: HiddenState, action: Action) -> HiddenState:
    """Stub enactor: no-op; real handlers wire to TF / probes / dataset loaders."""
    log.info("enact %s payload=%s", action.kind.value, action.payload)
    return state


class PhenotypingAgent:
    """Self-evidencing loop. Accepts a specimen when surprise drops below tau."""

    def __init__(
        self,
        catalogue: Catalogue,
        preferences: Optional[Preferences] = None,
        observer: Observer = default_observer,
        policies: PolicyEnumerator = default_policies,
        enactor: Enactor = default_enactor,
        accept_tau: float = 0.5,
    ):
        self.catalogue = catalogue
        self.preferences = preferences or Preferences()
        self.observer = observer
        self.policies = policies
        self.enactor = enactor
        self.accept_tau = accept_tau

    def expected_free_energy(self, state: HiddenState, action: Action) -> float:
        """G(pi) = pragmatic + epistemic.

        Pragmatic: predicted surprise after enacting `action`.
        Epistemic: rough novelty bonus that prefers non-REFRAIN exploration.
        Replace both with proper variational estimates.
        """
        predicted = self.observer(state)
        pragmatic = self.preferences.surprise(predicted, state.bio)
        epistemic = 0.0 if action.kind == ActionKind.REFRAIN else -0.1
        return pragmatic + epistemic

    def step(self, state: HiddenState) -> HiddenState:
        actions = list(self.policies(state))
        scored = [(self.expected_free_energy(state, a), a) for a in actions]
        scored.sort(key=lambda x: x[0])
        chosen_g, chosen = scored[0]
        log.info("argmin_pi G = %.4f via %s", chosen_g, chosen.kind.value)

        new_state = self.enactor(state, chosen)
        obs = self.observer(new_state)
        F = self.preferences.surprise(obs, new_state.bio)

        if F < self.accept_tau:
            self.catalogue.append(
                Specimen(
                    arch_spec=asdict(new_state.arch),
                    training={"data": asdict(new_state.data), "seed": new_state.theta.seed},
                    capability={
                        "accuracy": new_state.cap.accuracy,
                        "ece": new_state.cap.calibration_ece,
                        "ood_accuracy": new_state.cap.ood_accuracy,
                        "shape_bias": new_state.cap.shape_bias,
                    },
                    plausibility=asdict(new_state.bio),
                    neural_alignment=dict(new_state.cap.neural_alignment),
                    free_energy=F,
                    note=f"accepted via {chosen.kind.value}",
                )
            )
        return new_state

    def run(self, state: HiddenState, n_steps: int = 1) -> HiddenState:
        for _ in range(n_steps):
            state = self.step(state)
        return state
