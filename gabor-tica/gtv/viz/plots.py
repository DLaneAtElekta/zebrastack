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
