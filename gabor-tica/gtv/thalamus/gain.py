"""Thalamic gain / attention field (plan Phase 6).

Templates come from the generative path: clamp the top (AIT) latents to a
category and decode downward. A stage's attention field is a gain on its
Gabor energies, applied before divisive normalization (``HigherStage``'s
``gain`` argument), i.e. the normalization model of attention
(Reynolds & Heeger 2009): drive = A * E, R = drive / (sigma + pool(drive)).
"""

import torch


def feature_gain(templates: torch.Tensor, target: int, beta: float = 1.0) -> torch.Tensor:
    """Per-channel gain A = exp(beta * z) from category templates (K, C): z is
    how far the target's template exceeds the mean template on each channel,
    in units of the across-category spread. beta = 0 gives A = 1 (no attention)."""
    z = (templates[target] - templates.mean(0)) / (templates.std(0) + 1e-6)
    return torch.exp(beta * z)


def feature_similarity_field(stage, x: torch.Tensor, templates: torch.Tensor, target: int,
                             beta: float = 1.0) -> torch.Tensor:
    """Spatial attention field from feature similarity (Treue & Martinez-Trujillo):
    A(x, y) = exp(beta * m(x, y)), where m is the per-image z-scored match
    between the stage's local second-order features (log-normalized Gabor
    energies, before pooling) and the target template's deviation from the
    mean template. Locations that look like the target are boosted, clutter
    suppressed, without knowing where the target is. Returns a gain of shape
    (B, 1, H, W) for ``HigherStage(..., gain=...)``.
    """
    from gtv.stages.gabor import energy

    z = (templates[target] - templates.mean(0)) / (templates.std(0) + 1e-6)
    with torch.no_grad():
        f = stage.log(stage.norm(energy(stage.bank(x))))  # (B, n_second, H, W)
        f = f - f.mean(1, keepdim=True)
        m = torch.einsum("c,bchw->bhw", z, f)
        m = (m - m.mean((1, 2), keepdim=True)) / (m.std((1, 2), keepdim=True) + 1e-6)
    return torch.exp(beta * m).unsqueeze(1)
