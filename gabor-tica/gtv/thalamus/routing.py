"""Pulvinar-style routing (plan Phase 8): select which region feeds the stack.

A shifter gate (after Olshausen, Anderson & Van Essen 1993) in its simplest
form. A priority map over a stage's sheet chooses one location; a window
around it is cut from the input, rescaled to the scale the stack was fitted
on, and centered, so the next pass processes that region object-centered
instead of pooling it with the rest of the scene.

Priority maps:
- template match: how well the stage's local second-order features match
  the target template's deviation from the mean template (attention-guided);
- saliency: the stage's local second-order energy, whatever the features;
- learned: a linear per-cell detector of the target on the stage's features,
  trained on scenes with known item positions (Phase 8b).
With ``select_topk`` the gate visits several locations in turn (serial
glimpses), each processed object-centered.
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


def _masked(priority: torch.Tensor, border: int) -> torch.Tensor:
    if not border:
        return priority
    p = priority.clone()
    p[:, :border], p[:, -border:], p[:, :, :border], p[:, :, -border:] = (-float("inf"),) * 4
    return p


def select_location(priority: torch.Tensor, image_size: int, border: int = 0) -> torch.Tensor:
    """(B, H, W) priority map -> (B, 2) pixel center (y, x) of its argmax cell.
    ``border``: ignore this many cells at the edge (padding artifacts; the
    stages are fitted without them)."""
    return select_topk(priority, image_size, 1, border)[:, 0]


def select_topk(priority: torch.Tensor, image_size: int, k: int, border: int = 0,
                min_sep: int = 2) -> torch.Tensor:
    """(B, H, W) -> (B, k, 2) pixel centers of the k highest cells, greedily
    suppressing cells within ``min_sep`` (Chebyshev, in cells) of a chosen one
    (serial glimpses that do not revisit the same object)."""
    b, h, w = priority.shape
    p = _masked(priority, border).clone()
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    out = []
    for _ in range(k):
        idx = p.flatten(1).argmax(1)
        cy, cx = idx // w, idx % w
        out.append(torch.stack([(cy.float() + 0.5) * image_size / h, (cx.float() + 0.5) * image_size / w], 1))
        near = ((yy.view(1, h, w) - cy.view(-1, 1, 1)).abs() < min_sep) & \
               ((xx.view(1, h, w) - cx.view(-1, 1, 1)).abs() < min_sep)
        p = p.masked_fill(near, -float("inf"))
    return torch.stack(out, 1)


def learned_priority(features: torch.Tensor, weight: torch.Tensor, bias: float,
                     mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H, W) logit of a linear per-cell detector (trained on
    standardized cell feature vectors, e.g. in experiments/phase8_routing.py)."""
    z = (features - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    return torch.einsum("c,bchw->bhw", weight, z) + bias


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
