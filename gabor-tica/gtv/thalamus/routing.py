"""Pulvinar-style routing (plan Phase 8): select which region feeds the stack.

A shifter gate (after Olshausen, Anderson & Van Essen 1993) in its simplest
form. A priority map over a stage's sheet chooses one location; a window
around it is cut from the input, rescaled to the scale the stack was fitted
on, and centered, so the next pass processes that region object-centered
instead of pooling it with the rest of the scene.

Priority maps:
- template match: how well the stage's local second-order features match
  the target template's deviation from the mean template (attention-guided);
- saliency: the stage's local second-order energy, whatever the features.
"""

import torch
import torch.nn.functional as F


def template_match_map(features: torch.Tensor, templates: torch.Tensor, target: int) -> torch.Tensor:
    """(B, C, H, W) second-order features -> (B, H, W) match to the target's
    template deviation (projection on its direction, relative to the mean template)."""
    mean = templates.mean(0)
    d = templates[target] - mean
    u = d / (d.norm() + 1e-8)
    return torch.einsum("c,bchw->bhw", u, features - mean.view(1, -1, 1, 1))


def saliency_map(features: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H, W): local feature energy relative to the image mean."""
    f = features - features.mean((2, 3), keepdim=True)
    return f.pow(2).sum(1)


def select_location(priority: torch.Tensor, image_size: int) -> torch.Tensor:
    """(B, H, W) priority map -> (B, 2) pixel center (y, x) of its argmax cell."""
    b, h, w = priority.shape
    idx = priority.flatten(1).argmax(1)
    cy, cx = idx // w, idx % w
    return torch.stack([(cy.float() + 0.5) * image_size / h, (cx.float() + 0.5) * image_size / w], 1)


def route_window(images: torch.Tensor, centers: torch.Tensor, window: int = 28, scale: float = 2.0,
                 out_size: int | None = None) -> torch.Tensor:
    """Cut a ``window`` x ``window`` region around each center (clamped inside the
    image) from raw images (B, 1, S, S) or (B, S, S), rescale it by ``scale``
    and center it on an ``out_size`` canvas (default S), zero-mean.
    Returns (B, 1, out_size, out_size)."""
    if images.dim() == 3:
        images = images.unsqueeze(1)
    b, _, s, _ = images.shape
    out_size = out_size or s
    half = window // 2
    out = []
    for i in range(b):
        y0 = int(round(float(centers[i, 0]))) - half
        x0 = int(round(float(centers[i, 1]))) - half
        y0, x0 = min(max(y0, 0), s - window), min(max(x0, 0), s - window)
        crop = images[i : i + 1, :, y0 : y0 + window, x0 : x0 + window]
        crop = crop - crop.amin()  # the scene is zero-mean; restore a black background
        up = F.interpolate(crop, scale_factor=scale, mode="bilinear", align_corners=False)
        pad = (out_size - up.shape[-1]) // 2
        canvas = F.pad(up, (pad, out_size - up.shape[-1] - pad, pad, out_size - up.shape[-1] - pad))
        out.append(canvas - canvas.mean())
    return torch.cat(out)
