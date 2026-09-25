"""Temporal coherence for TICA (plan Phase 4).

Two objectives on pairs of whitened inputs (z_t, z_t1) from consecutive frames,
both with orthonormal W (Riemannian gradient descent, as in complete TICA):

  bubbles   (Hyvarinen, Hurri & Vayrynen 2003) pool each unit's energy over its
            sheet neighborhood AND the two frames:
                sum_i sqrt( sum_j h_ij (s_j(t)^2 + s_j(t+1)^2) + eps )
            sqrt(a + b) <= sqrt(a) + sqrt(b), so energy that persists in the same
            pool across frames costs less than energy that hops between pools.
            ``temporal_weight`` = 0 recovers still-image TICA on both frames.
  coherence still-image TICA minus lam * mean sum_i s_i(t)^2 s_i(t+1)^2: temporal
            coherence of activity levels (Hurri & Hyvarinen 2003). With
            orthonormal W, E[s_i^2] = 1 is fixed, so this rewards energy that
            persists across frames.

Two "slowness" penalties were tried first and do not work with a complete
orthonormal W: linear slowness sum_i E[(s_i(t+1) - s_i(t))^2] = trace(W C W^T)
is the same for every rotation, and an unnormalized penalty on pooled-energy
differences is minimized by making energies less variable (mixing sources),
not by making them persist.
"""

import torch

from gtv.stages.tica import arrange_on_sheet, symmetric_orthonormalize, tica_loss


def bubbles_loss(s_t: torch.Tensor, s_t1: torch.Tensor, h: torch.Tensor, temporal_weight: float = 1.0,
                 eps: float = 1e-3) -> torch.Tensor:
    """Mean over samples of the spatiotemporally pooled TICA energy.

    With weight w the pool for frame t is h(s_t^2 + w s_t1^2) and symmetrically
    for t+1; w = 0 gives the still loss summed over both frames."""
    e_t, e_t1 = s_t.pow(2) @ h.T, s_t1.pow(2) @ h.T
    return (torch.sqrt(e_t + temporal_weight * e_t1 + eps) + torch.sqrt(e_t1 + temporal_weight * e_t + eps)).sum(1).mean()


def coherence_loss(s_t: torch.Tensor, s_t1: torch.Tensor, h: torch.Tensor, lam: float = 1.0,
                   eps: float = 1e-3) -> torch.Tensor:
    """Still TICA on both frames minus lam * E[sum_i s_i(t)^2 s_i(t+1)^2]."""
    coherence = (s_t.pow(2) * s_t1.pow(2)).sum(1).mean()
    return tica_loss(s_t, h, eps) + tica_loss(s_t1, h, eps) - lam * coherence


def fit_temporal(
    z_t: torch.Tensor,
    z_t1: torch.Tensor,
    h: torch.Tensor,
    mode: str = "bubbles",
    weight: float = 1.0,
    n_iter: int = 500,
    lr: float = 0.5,
    eps: float = 1e-3,
    arrange: bool = True,
    polish_iter: int = 200,
    seed: int = 0,
) -> tuple[torch.Tensor, list[float]]:
    """Orthonormal W (d, d) minimizing the chosen temporal objective on whitened
    pairs. ``mode``: "bubbles" (weight = temporal pooling weight), "coherence"
    (weight = lam), or "still" (TICA on each frame independently)."""
    d = z_t.shape[1]

    def loss_fn(w):
        s_t, s_t1 = z_t @ w.T, z_t1 @ w.T
        if mode == "bubbles":
            return bubbles_loss(s_t, s_t1, h, weight, eps)
        if mode == "coherence":
            return coherence_loss(s_t, s_t1, h, weight, eps)
        if mode == "still":
            return bubbles_loss(s_t, s_t1, h, 0.0, eps)
        raise ValueError(f"unknown mode {mode!r}")

    def descend(w, iters):
        hist = []
        for _ in range(iters):
            w.requires_grad_(True)
            loss = loss_fn(w)
            (grad,) = torch.autograd.grad(loss, w)
            with torch.no_grad():
                w = symmetric_orthonormalize(w - lr * (grad - w @ grad.T @ w))
            hist.append(loss.item())
        return w, hist

    gen = torch.Generator().manual_seed(seed)
    w, hist = descend(symmetric_orthonormalize(torch.randn(d, d, generator=gen)), n_iter)
    if arrange:
        s = torch.cat([z_t @ w.T, z_t1 @ w.T])
        w = w[arrange_on_sheet(s.detach(), h)]
        w, more = descend(w, polish_iter)
        hist += more
    return w.detach(), hist
