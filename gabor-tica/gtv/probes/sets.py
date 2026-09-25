"""Labelled probe sets for linear-readout comparisons (Phase 2 onward).

Each builder takes a probe config dict ``p`` (see ``configs/phase2.yaml``,
``probes:``), the image size and a generator, and returns images of shape
(N, 1, size, size) with labels.
"""

import math

import torch

from .stimuli import JUNCTION_ARMS, contrast_modulated_grating, junction, oriented_noise, texture_pair


def rand(gen, lo=0.0, hi=1.0):
    return lo + (hi - lo) * float(torch.rand(1, generator=gen))


# ---------------------------------------------------------------- probe sets
def texture_set(p, size, gen):
    """First-order: 8 orientation classes of band-pass noise."""
    imgs, labels = [], []
    for k in range(8):
        for _ in range(p["per_class"]):
            imgs.append(oriented_noise(size, k * math.pi / 8, p["texture_freq"], p["texture_bandwidth"], gen))
            labels.append(k)
    return torch.stack(imgs).unsqueeze(1), torch.tensor(labels)


def second_order_set(p, size, gen):
    """Second-order: 4 envelope-orientation classes; carrier orientation and
    envelope phase random, so mean V1 energy carries no class information."""
    imgs, labels = [], []
    for k in range(4):
        for _ in range(p["per_class"]):
            imgs.append(contrast_modulated_grating(
                size, p["cm_envelope_freq"], k * math.pi / 4,
                carrier_theta=rand(gen, 0, math.pi), carrier_freq=p["cm_carrier_freq"],
                generator=gen, envelope_phase=rand(gen, 0, 2 * math.pi)))
            labels.append(k)
    return torch.stack(imgs).unsqueeze(1), torch.tensor(labels)


def junction_set(p, size, gen):
    kinds = list(JUNCTION_ARMS)
    imgs, labels = [], []
    for k, kind in enumerate(kinds):
        for _ in range(p["per_class"]):
            c = (rand(gen, -2, 2), rand(gen, -2, 2))
            imgs.append(junction(size, kind, rand(gen, 0, 2 * math.pi), center=c))
            labels.append(k)
    return torch.stack(imgs).unsqueeze(1), torch.tensor(labels), kinds


def boundary_set(p, size, gen, cell=4, border=2):
    """Texture pairs with random orientations/boundary; per-cell labels:
    1 if the boundary passes within 3 px of the cell center, 0 if > 10 px."""
    imgs, dists = [], []
    grid = size // cell
    cc = (torch.arange(grid) + 0.5) * cell - size / 2
    cy, cx = torch.meshgrid(cc, cc, indexing="ij")
    for _ in range(p["boundary_images"]):
        ta = rand(gen, 0, math.pi)
        tb = ta + rand(gen, math.pi / 4, 3 * math.pi / 4)
        bt, off = rand(gen, 0, math.pi), rand(gen, -12, 12)
        img, _ = texture_pair(size, ta, tb, bt, p["texture_freq"], gen, off)
        imgs.append(img)
        dists.append((cx * math.cos(bt) + cy * math.sin(bt) - off).abs())
    d = torch.stack(dists)
    inner = torch.zeros(grid, grid, dtype=torch.bool)
    inner[border:-border, border:-border] = True
    return torch.stack(imgs).unsqueeze(1), d, inner


SCENE_KINDS = ("target", "distractor", "absent")


def second_order_scene(
    size: int,
    background: torch.Tensor,
    kind: str,
    gen: torch.Generator,
    target_theta: float = math.pi / 4,
    distractor_theta: float = 3 * math.pi / 4,
    envelope_freq: float = 0.0625,
    carrier_freq: float = 0.25,
    target_contrast: float = 1.0,
    window_sigma: float = 12.0,
) -> torch.Tensor:
    """A texture patch blended into natural clutter (the V2-level analog of a
    target in a cluttered scene, for FE-3).

    ``target``: carrier contrast-modulated at ``target_theta``; ``distractor``:
    modulated at ``distractor_theta``; ``absent``: the same carrier,
    unmodulated. Carrier orientation and envelope phase are random, so only the
    second-order structure separates target from absent. ``background`` is a
    (size, size) clutter patch, scaled here to unit standard deviation.
    """
    if kind not in SCENE_KINDS:
        raise ValueError(f"kind must be one of {SCENE_KINDS}")
    bg = (background - background.mean()) / (background.std() + 1e-8)
    carrier = oriented_noise(size, rand(gen, 0, math.pi), carrier_freq, generator=gen)
    carrier = carrier / (carrier.std() + 1e-8)
    if kind == "absent":
        envelope = torch.full((size, size), 0.5)
    else:
        theta = target_theta if kind == "target" else distractor_theta
        from .stimuli import grating

        envelope = 0.5 * (1 + grating(size, envelope_freq, theta, rand(gen, 0, 2 * math.pi)))
    # equal mean power in all kinds: E[envelope^2] is 3/8 when modulated, 1/4 when flat
    if kind == "absent":
        envelope = envelope * math.sqrt(1.5)
    c = (torch.arange(size, dtype=torch.float32) - (size - 1) / 2)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    window = torch.exp(-0.5 * (xx**2 + yy**2) / window_sigma**2)
    return bg + target_contrast * window * carrier * envelope


def clutter_scene(items: list[torch.Tensor], size: int, gen: torch.Generator) -> torch.Tensor:
    """Place 28x28 item images (values >= 0, black background) at random
    non-clipped positions on a size x size canvas, combined by max (nearer
    items occlude); returns the zero-mean canvas (size, size)."""
    canvas = torch.zeros(size, size)
    for it in items:
        h, w = it.shape
        y = int(torch.randint(0, size - h + 1, (1,), generator=gen))
        x = int(torch.randint(0, size - w + 1, (1,), generator=gen))
        canvas[y : y + h, x : x + w] = torch.maximum(canvas[y : y + h, x : x + w], it)
    return canvas - canvas.mean()
