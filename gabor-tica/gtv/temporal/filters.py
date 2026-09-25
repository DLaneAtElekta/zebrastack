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

import torch

from gtv.temporal.coherence import bubbles_loss


def _vectors(f: torch.Tensor, border: int) -> torch.Tensor:
    if border:
        f = f[..., border:-border, border:-border]
    return f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])


def fit_stage_filters(
    stage,
    x_t: torch.Tensor,
    x_t1: torch.Tensor,
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
) -> dict[str, list[float]]:
    """Train ``stage.bank`` on input-map pairs ``x_t``, ``x_t1`` (N, C, H, W).

    ``lr`` is relative to the initial kernels' rms (Adam steps are about
    lr * rms per element). Whitening + TICA are refit (on both frames) every
    ``refit_every`` steps, on the first ``refit_n`` pairs; the caller refits
    once more on its own data after.
    Returns the loss history by term and the filter drift."""
    tica_kw = tica_kw or {}
    gen = torch.Generator().manual_seed(seed)
    bank = stage.bank
    opt = torch.optim.Adam([bank.weight], lr=lr * float(bank.weight_init.pow(2).mean().sqrt()))
    both = torch.cat([x_t[:refit_n], x_t1[:refit_n]])
    hist = {"bubbles": [], "white": [], "tether": [], "drift": []}
    n_units = stage.tica.n_units
    eye = torch.eye(n_units)
    for step in range(n_steps):
        if step % refit_every == 0:
            stage.fit(both, border=border, seed=seed, **tica_kw)
            m, b = (t.detach() for t in stage.tica.affine)
            h = stage.tica.h
        idx = torch.randint(len(x_t), (batch_size,), generator=gen)
        s_t = _vectors(stage.features(x_t[idx]), border) @ m.T + b
        s_t1 = _vectors(stage.features(x_t1[idx]), border) @ m.T + b
        sparse = bubbles_loss(s_t, s_t1, h, temporal_weight, stage.tica.eps) / n_units
        s = torch.cat([s_t, s_t1])
        s = s - s.mean(0)
        white = (s.T @ s / len(s) - eye).pow(2).sum() / n_units
        teth = bank.tether()
        loss = sparse + white_weight * white + tether * teth
        opt.zero_grad()
        loss.backward()
        opt.step()
        for k, v in (("bubbles", sparse), ("white", white), ("tether", teth)):
            hist[k].append(float(v.detach()))
        hist["drift"].append(bank.drift())
    return hist
