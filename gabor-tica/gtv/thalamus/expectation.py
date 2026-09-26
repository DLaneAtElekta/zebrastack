"""Expectation channel (plan Phase 7): predictive subtraction.

Attention (``gain.py``) scales a stage's energies; expectation subtracts the
predicted part of its output, so what the stage passes up is prediction
error. The prediction for an expected category is its template's deviation
from the mean template, d = t_target - mean_k t_k, in the stage's
second-order feature units (log-normalized energies). It is placed where the
local input matches it:

    L'(x, y) = L(x, y) - alpha * w(x, y) * d,   w = relu(z-scored match of L to d)

so input that looks like the expectation is explained away, and what differs
from it (a lookalike's mismatching features) stays as error. The match is
computed from the signal the stage actually passes up (after the response
noise), so no clean information bypasses the bottleneck. ``spatial=False``
subtracts alpha * d everywhere: a constant offset per channel, the control
that should carry no information.

With ``predict`` set to a superordinate group, the prediction is the group's
shared pattern (Phase 7b): a hierarchical expectation that removes what the
target shares with its lookalikes.
"""

from typing import Callable

import torch


def expectation(templates: torch.Tensor, target: int, alpha: float = 1.0,
                spatial: bool = True, predict: list[int] | None = None,
                match: str = "zscore") -> Callable[[torch.Tensor], torch.Tensor]:
    """Predictive subtraction for ``HigherStage(..., expect=...)``: a function
    mapping log-normalized energies L (B, C, H, W) to prediction errors.
    ``predict``: the categories whose mean template is predicted (a
    superordinate expectation, e.g. all footwear); default the target alone.
    Explaining away a group's shared component leaves what is specific to each
    member (e.g. sneaker vs sandal) in the error.
    ``match``: how much of the prediction is placed at each location.
    "zscore" (Phase 7): the per-image z-scored match, rectified; it grows with
    how isolated a match is, so it can subtract more than is there.
    "projection": the rectified regression coefficient of the local deviation
    on d, i.e. the amount of the predicted pattern present; alpha = 1 then
    removes exactly the matched component."""
    mean = templates.mean(0)
    d = (templates[target] if predict is None else templates[predict].mean(0)) - mean
    u = d / (d.norm() + 1e-8)

    def explain_away(L: torch.Tensor) -> torch.Tensor:
        if not spatial:
            return L - alpha * d.view(1, -1, 1, 1)
        m = torch.einsum("c,bchw->bhw", u, L - mean.view(1, -1, 1, 1))
        if match == "zscore":
            m = (m - m.mean((1, 2), keepdim=True)) / (m.std((1, 2), keepdim=True) + 1e-6)
        elif match == "projection":
            m = m / (d.norm() + 1e-8)
        else:
            raise ValueError(f"unknown match {match!r}")
        w = m.clamp(min=0).unsqueeze(1)
        return L - alpha * w * d.view(1, -1, 1, 1)

    return explain_away
