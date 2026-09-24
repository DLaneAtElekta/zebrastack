"""Matplotlib diagnostics. Each function returns a Figure; callers save it."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def _to_2d_list(images: torch.Tensor) -> list:
    """Accept (N, H, W) or (N, 1, H, W)."""
    if images.dim() == 4:
        if images.shape[1] != 1:
            raise ValueError("expected single-channel images, got shape %s" % (tuple(images.shape),))
        images = images[:, 0]
    if images.dim() != 3:
        raise ValueError("expected (N, H, W) or (N, 1, H, W), got %s" % (tuple(images.shape),))
    return [img.numpy() for img in images.detach().cpu()]


def _grid(images: list, ncols: int | None, titles, cmap, symmetric) -> Figure:
    n = len(images)
    ncols = ncols or math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.6 * nrows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= n:
            continue
        img = images[i]
        if symmetric:
            lim = float(abs(img).max()) or 1.0
            ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim)
        else:
            ax.imshow(img, cmap=cmap)
        if titles is not None:
            ax.set_title(titles[i], fontsize=7)
    fig.tight_layout()
    return fig


def filter_atlas(filters: torch.Tensor, ncols: int | None = None, titles=None) -> Figure:
    """Grid of filters (N, H, W), each scaled symmetrically around zero."""
    return _grid(_to_2d_list(filters), ncols, titles, "RdBu_r", symmetric=True)


def fantasy_grid(samples: torch.Tensor, ncols: int | None = None, titles=None) -> Figure:
    """Grid of generated samples or stage outputs (N, H, W)."""
    return _grid(_to_2d_list(samples), ncols, titles, "gray", symmetric=False)


def sheet_map(
    preference: torch.Tensor,
    strength: torch.Tensor | None = None,
    cyclic: bool = True,
    title: str = "",
) -> Figure:
    """Topographic sheet colored by each unit's preferred feature.

    ``preference`` is (H, W) over the TICA sheet, e.g. preferred orientation in
    [0, pi). ``cyclic`` picks a cyclic colormap so pinwheels read correctly.
    ``strength`` (H, W), if given, sets per-unit opacity (e.g. selectivity).
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    pref = preference.detach().cpu().numpy()
    im = ax.imshow(pref, cmap="hsv" if cyclic else "viridis", interpolation="nearest")
    if strength is not None:
        s = strength.detach().cpu()
        im.set_alpha((s / (s.max() + 1e-8)).clamp(0, 1).numpy())
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def orientation_energy_image(v1_map: torch.Tensor, n_orientations: int, n_scales: int, scale: int | None = None):
    """RGB rendering of one V1 log-energy map (C, H, W), C = scales x orientations.

    Hue = dominant orientation (doubled-angle vector average of energies),
    saturation = orientation selectivity, value = total energy. ``scale`` picks
    one scale; None sums energies over scales.
    """
    import numpy as np
    from matplotlib.colors import hsv_to_rgb

    e = v1_map.detach().cpu().double().exp().view(n_scales, n_orientations, *v1_map.shape[-2:])
    e = e[scale] if scale is not None else e.sum(0)
    ang = torch.arange(n_orientations, dtype=torch.float64) * 2 * math.pi / n_orientations
    vec = (e * torch.exp(1j * ang).view(-1, 1, 1)).sum(0)
    total = e.sum(0)
    hue = torch.remainder(torch.angle(vec), 2 * math.pi) / (2 * math.pi)
    sat = (vec.abs() / total.clamp(min=1e-12)).clamp(0, 1)
    lv = total.log()
    val = ((lv - lv.min()) / (lv.max() - lv.min() + 1e-12)).clamp(0, 1)
    return hsv_to_rgb(np.stack([hue.numpy(), sat.numpy(), val.numpy()], -1))


def orientation_grid(rows: dict[str, torch.Tensor], n_orientations: int, n_scales: int, ncols: int = 6,
                     scale: int | None = None) -> Figure:
    """One row of orientation-energy renderings per named batch of V1 maps."""
    fig, axes = plt.subplots(len(rows), ncols, figsize=(1.5 * ncols, 1.6 * len(rows)), squeeze=False)
    for r, (name, maps) in enumerate(rows.items()):
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(maps):
                ax.imshow(orientation_energy_image(maps[c], n_orientations, n_scales, scale), interpolation="nearest")
        axes[r, 0].set_title(name, fontsize=8, loc="left")
    fig.tight_layout()
    return fig
