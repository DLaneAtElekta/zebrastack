"""Temporal-coherence learning of an upper stage's Gabor-initialized filters.

Phase 4 learned only the rotation W on top of fixed features, which cannot
create new features. Here the stage's second-order filters themselves
(``LearnedHigherStage``, a ``SpatialGaborBank``) are trained by gradient
descent through energy -> normalization -> log -> pooling, with the stage's
whitening + TICA held fixed between periodic closed-form refits:

    loss = bubbles(s_t, s_t+1) / n_units               sparsity + temporal pooling
         + white_weight * ||cov(s) - I||_F^2 / n_units  keep outputs white
         + tether * ||K - K_gabor||^2 / ||K_gabor||^2   stay near the Gabors

``s`` are the stage's outputs at matching locations in two frames. With
``temporal_weight`` = 0 the first term is still-image TICA on each frame (the
control: filters learned for sparsity alone). The whiteness term stands in for
RICA's reconstruction cost (for white inputs the two agree): without it the
filters could lower the sparsity cost by shrinking or duplicating their
outputs, and slowness alone favors uninformative low-frequency blobs.
"""

import math

import torch

from gtv.temporal.coherence import bubbles_seq_loss


def _vectors(f: torch.Tensor, border: int) -> torch.Tensor:
    if border:
        f = f[..., border:-border, border:-border]
    return f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])


def fit_stage_filters(
    stage,
    x_t: torch.Tensor,
    x_t1: torch.Tensor | None = None,
    temporal_weight: float = 1.0,
    tether: float = 0.1,
    white_weight: float = 1.0,
    n_steps: int = 600,
    lr: float = 0.01,
    batch_size: int = 32,
    refit_every: int = 200,
    refit_n: int | None = 500,
    border: int = 0,
    seed: int = 0,
    tica_kw: dict | None = None,
    optimizer: str = "adam",
    momentum: float = 0.9,
) -> dict[str, list[float]]:
    """Train ``stage.bank`` on input-map pairs ``x_t``, ``x_t1`` (N, C, H, W),
    or on sequences: ``x_t`` (N, T, C, H, W) and ``x_t1`` None (the bubbles
    pool then spans all T frames, ``bubbles_seq_loss``).

    ``lr`` is relative to ``bank.step_scale()`` (the initial kernels' rms for
    spatial kernels, 1 for Gabor-mixing coefficients).

    ``optimizer``: "adam" (Phase 5c) or "ngd", normalized gradient descent
    with momentum and a cosine-decayed step: each step moves the weights by
    lr * ||K_init|| along the momentum of unit-norm gradients. Unlike Adam it
    keeps the gradient's relative sizes across elements, so weights with tiny
    gradients (kernel periphery, unused mixing terms) barely move. Whitening + TICA are refit (on both frames) every
    ``refit_every`` steps, on the first ``refit_n`` pairs; the caller refits
    once more on its own data after.
    Returns the loss history by term and the filter drift."""
    tica_kw = tica_kw or {}
    gen = torch.Generator().manual_seed(seed)
    bank = stage.bank
    if optimizer == "adam":
        opt = torch.optim.Adam([bank.weight], lr=lr * bank.step_scale())
    elif optimizer == "ngd":
        step_len = lr * float(bank.weight_init.norm())
        vel = torch.zeros_like(bank.weight)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    seq = torch.stack([x_t, x_t1], 1) if x_t1 is not None else x_t  # (N, T, C, H, W)
    both = seq[:refit_n].flatten(0, 1)
    hist = {"bubbles": [], "white": [], "tether": [], "drift": []}
    n_units = stage.tica.n_units
    eye = torch.eye(n_units)
    for step in range(n_steps):
        if step % refit_every == 0:
            stage.fit(both, border=border, seed=seed, **tica_kw)
            m, b = (t.detach() for t in stage.tica.affine)
            h = stage.tica.h
        idx = torch.randint(len(seq), (batch_size,), generator=gen)
        xb = seq[idx]  # (B, T, C, H, W)
        f = stage.features(xb.flatten(0, 1))
        f = f.view(len(idx), -1, *f.shape[1:])
        frames = [_vectors(f[:, t], border) @ m.T + b for t in range(f.shape[1])]
        sparse = bubbles_seq_loss(frames, h, temporal_weight, stage.tica.eps) / n_units
        s = torch.cat(frames)
        s = s - s.mean(0)
        white = (s.T @ s / len(s) - eye).pow(2).sum() / n_units
        teth = bank.tether()
        loss = sparse + white_weight * white + tether * teth
        if optimizer == "adam":
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            (grad,) = torch.autograd.grad(loss, bank.weight)
            with torch.no_grad():
                vel.mul_(momentum).add_(grad / (grad.norm() + 1e-12))
                bank.weight.sub_(step_len * 0.5 * (1 + math.cos(math.pi * step / n_steps)) * vel)
        for k, v in (("bubbles", sparse), ("white", white), ("tether", teth)):
            hist[k].append(float(v.detach()))
        hist["drift"].append(bank.drift())
    return hist
