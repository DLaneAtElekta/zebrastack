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
"""

from typing import Callable

import torch


def expectation(templates: torch.Tensor, target: int, alpha: float = 1.0,
                spatial: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    """Predictive subtraction for ``HigherStage(..., expect=...)``: a function
    mapping log-normalized energies L (B, C, H, W) to prediction errors."""
    mean = templates.mean(0)
    d = templates[target] - mean
    u = d / (d.norm() + 1e-8)

    def explain_away(L: torch.Tensor) -> torch.Tensor:
        if not spatial:
            return L - alpha * d.view(1, -1, 1, 1)
        m = torch.einsum("c,bchw->bhw", u, L - mean.view(1, -1, 1, 1))
        m = (m - m.mean((1, 2), keepdim=True)) / (m.std((1, 2), keepdim=True) + 1e-6)
        w = m.clamp(min=0).unsqueeze(1)
        return L - alpha * w * d.view(1, -1, 1, 1)

    return explain_away
