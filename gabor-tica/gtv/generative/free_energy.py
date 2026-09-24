"""Unified variational free energy, step FE-1 (plan Section 7).

One objective, minimized over encoder and decoder together on real data (no
fantasies, so no sleep-phase bias):

    F = 1/2 pi ||x - g(z)||^2 - N/2 log pi          accuracy, precision pi fixed (FE-1)
      + sum_locations sum_i sqrt(sum_j h_ij z_j^2 + eps)   TICA prior on the V2 sheet
      - H[q(z | x)]                                  entropy of the Gaussian posterior

    (plus constants), with z ~ q by reparameterization. q(z | x) = N(mu(x),
    diag sigma(x)^2): the fixed Design B front end followed by learnable
    affine heads for mu and log sigma. g is a ``ConvFactorAnalysis`` mean.

FE-2 makes pi learnable per channel; FE-3 conditions it on context.
"""

import math

import torch
from torch import nn

from gtv.stages.tica import arrange_on_sheet
from gtv.stages.v2 import V2Stage

from .decoders import ConvFactorAnalysis


class TICAPrior(nn.Module):
    """-log p(z) = sum over locations of sum_i sqrt(sum_j h_ij z_j^2 + eps), up to log Z.

    Latents are independent across map locations; the sheet neighborhood h
    couples units at the same location, as in Phase 2."""

    def __init__(self, h: torch.Tensor, eps: float = 1e-3):
        super().__init__()
        self.register_buffer("h", h)
        self.eps = eps

    def neg_log(self, z: torch.Tensor) -> torch.Tensor:
        """(B, n, H, W) -> (B,) summed over units and locations."""
        pooled = torch.einsum("ij,bjhw->bihw", self.h, z.pow(2))
        return torch.sqrt(pooled + self.eps).sum((1, 2, 3))


class AmortizedEncoder(nn.Module):
    """q(z | x) = N(mu, sigma^2) from affine heads on fixed Design B features."""

    def __init__(self, v2: V2Stage, init: str = "tica", init_sigma: float = 0.1, seed: int = 0):
        super().__init__()
        if v2.front is None:
            raise ValueError("the encoder expects a Design B V2Stage")
        self.front = v2.front
        m, b = v2.tica.affine
        if init == "random":
            # same whitening, random orthonormal rotation instead of the learned TICA one
            g = torch.Generator().manual_seed(seed)
            q, _ = torch.linalg.qr(torch.randn(m.shape[0], m.shape[0], generator=g))
            w = v2.tica.weight
            m, b = q @ (torch.linalg.pinv(w) @ m), q @ (torch.linalg.pinv(w) @ b)
        elif init != "tica":
            raise ValueError("init must be 'tica' or 'random'")
        n, d = m.shape
        self.mu_weight = nn.Parameter(m.clone())
        self.mu_bias = nn.Parameter(b.clone())
        self.ls_weight = nn.Parameter(torch.zeros(n, d))
        self.ls_bias = nn.Parameter(torch.full((n,), math.log(init_sigma)))

    def features(self, v1: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        with torch.no_grad():
            return torch.cat([self.front(x) for x in v1.split(chunk)])

    def from_features(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = torch.einsum("nd,bdhw->bnhw", self.mu_weight, f) + self.mu_bias.view(1, -1, 1, 1)
        log_sigma = torch.einsum("nd,bdhw->bnhw", self.ls_weight, f) + self.ls_bias.view(1, -1, 1, 1)
        return mu, log_sigma.clamp(-7, 2)

    def forward(self, v1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.from_features(self.features(v1))


def free_energy(
    x: torch.Tensor,
    f: torch.Tensor,
    encoder: AmortizedEncoder,
    decoder: ConvFactorAnalysis,
    prior: TICAPrior,
    log_precision: torch.Tensor,
    generator=None,
    border: int = 2,
) -> dict[str, torch.Tensor]:
    """Monte Carlo (one sample) free energy per image, split into terms.

    x: standardized V1 maps (B, C, H, W); f: encoder features of x.
    ``log_precision``: scalar or per-channel (C,) log pi. Border cells of the
    V1 map are excluded from the accuracy term (edge effects)."""
    mu, log_sigma = encoder.from_features(f)
    eps = torch.randn(mu.shape, generator=generator)
    z = mu + log_sigma.exp() * eps
    err = (x - decoder.mean(z))[..., border:-border, border:-border]
    lp = log_precision.view(1, -1, 1, 1) if log_precision.dim() else log_precision
    accuracy = (0.5 * err.pow(2) * lp.exp() - 0.5 * lp + 0.5 * math.log(2 * math.pi)).sum((1, 2, 3))
    complexity = prior.neg_log(z)
    entropy = (log_sigma + 0.5 * math.log(2 * math.pi * math.e)).sum((1, 2, 3))
    total = accuracy + complexity - entropy
    return {"F": total, "accuracy": accuracy, "prior": complexity, "entropy": entropy}


def rearrange_units(encoder: AmortizedEncoder, decoder: ConvFactorAnalysis, prior: TICAPrior,
                    f: torch.Tensor) -> torch.Tensor:
    """Re-seat latent units on the sheet so units with correlated energies are
    neighbors. F is invariant to permuting units except through the TICA prior,
    which this lowers; gradients alone cannot make such discrete moves (the
    Phase 2 lesson). Returns the permutation (sheet position -> old unit)."""
    with torch.no_grad():
        mu, _ = encoder.from_features(f)
        s = mu[..., 2:-2, 2:-2].permute(0, 2, 3, 1).reshape(-1, mu.shape[1])
        perm = arrange_on_sheet(s, prior.h)
        for p in (encoder.mu_weight, encoder.mu_bias, encoder.ls_weight, encoder.ls_bias):
            p.copy_(p[perm])
        decoder.weight.copy_(decoder.weight[:, perm])
    return perm


def fit_free_energy(
    v1_train: torch.Tensor,
    encoder: AmortizedEncoder,
    decoder: ConvFactorAnalysis,
    prior: TICAPrior,
    log_precision: torch.Tensor,
    n_iter: int = 1500,
    batch: int = 32,
    lr: float = 3e-3,
    warmup: int = 0,
    arrange_every: int | None = None,
    learn_precision: bool = False,
    free_precision_after: int = 0,
    log_precision_bounds: tuple[float, float] = (-2.0, 8.0),
    lr_precision: float = 1e-2,
    generator=None,
    log_every: int = 50,
) -> dict:
    """Minimize F jointly over encoder and decoder (amortized inference).

    ``warmup`` > 0 ramps the complexity terms (prior - entropy) in linearly
    over that many steps ("KL warm-up", plan 7.5), so units become useful for
    reconstruction before the prior can collapse them. ``arrange_every``
    re-seats units on the sheet at that interval (see ``rearrange_units``);
    Adam's moment estimates are reset after each re-seating.

    FE-2: with ``learn_precision`` the per-channel log precision (pass a (C,)
    tensor as the initial value) is learned by minimizing F too, but only after
    ``free_precision_after`` steps, and clamped to ``log_precision_bounds``
    (plan 7.5: runaway precision). The fitted value is returned in the history
    as ``log_precision``."""
    f_train = encoder.features(v1_train)
    x_train = decoder.from_v1(v1_train)
    params = [p for n, p in decoder.named_parameters() if n != "log_psi"] + list(encoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    if learn_precision:
        log_precision = nn.Parameter(log_precision.detach().clone().expand(x_train.shape[1]).contiguous())
        opt_pi = torch.optim.Adam([log_precision], lr=lr_precision)
    lo, hi = log_precision_bounds
    n_el = x_train[0, :, 2:-2, 2:-2].numel()
    hist = {"iter": [], "F_per_element": [], "accuracy": [], "prior": [], "entropy": []}
    for it in range(n_iter):
        if arrange_every and it > 0 and it % arrange_every == 0:
            sub = torch.randperm(len(f_train), generator=generator)[:128]
            rearrange_units(encoder, decoder, prior, f_train[sub])
            opt = torch.optim.Adam(params, lr=lr)  # moments belong to the old unit order
        idx = torch.randint(len(x_train), (batch,), generator=generator)
        terms = free_energy(x_train[idx], f_train[idx], encoder, decoder, prior, log_precision, generator)
        beta = min(1.0, it / warmup) if warmup else 1.0
        loss = (terms["accuracy"] + beta * (terms["prior"] - terms["entropy"])).mean() / n_el
        opt.zero_grad()
        if learn_precision:
            opt_pi.zero_grad()
        loss.backward()
        opt.step()
        if learn_precision and it >= free_precision_after:
            opt_pi.step()
            with torch.no_grad():
                log_precision.clamp_(lo, hi)
        if it % log_every == 0 or it == n_iter - 1:
            hist["iter"].append(it)
            hist["F_per_element"].append(terms["F"].mean().item() / n_el)
            for k in ("accuracy", "prior", "entropy"):
                hist[k].append(terms[k].mean().item() / n_el)
            hist.setdefault("log_precision_mean", []).append(float(log_precision.detach().mean()))
    hist["log_precision"] = log_precision.detach().clone()
    return hist


def residual_log_precision(encoder: AmortizedEncoder, decoder: ConvFactorAnalysis, v1: torch.Tensor,
                           border: int = 2) -> torch.Tensor:
    """Maximum-likelihood log precision per channel, log(1 / mean residual^2),
    using posterior samples (the accuracy term's own optimum)."""
    with torch.no_grad():
        x = decoder.from_v1(v1)
        mu, log_sigma = encoder(v1)
        z = mu + log_sigma.exp() * torch.randn_like(mu)
        err = (x - decoder.mean(z))[..., border:-border, border:-border]
        return -err.pow(2).mean((0, 2, 3)).log()


def encoder_r2(encoder: AmortizedEncoder, decoder: ConvFactorAnalysis, v1: torch.Tensor, border: int = 4) -> float:
    """Held-out R^2 of the V1 map reconstructed through the posterior mean."""
    with torch.no_grad():
        x = decoder.from_v1(v1)
        mu, _ = encoder(v1)
        xh = decoder.mean(mu)
        x, xh = x[..., border:-border, border:-border], xh[..., border:-border, border:-border]
        return float(1 - (x - xh).pow(2).mean() / x.var())


def active_units(encoder: AmortizedEncoder, v1: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Boolean (n,) mask of units whose posterior mean varies across data more
    than their posterior noise (std(mu) / mean(sigma) > threshold); the rest
    have collapsed onto the prior."""
    with torch.no_grad():
        mu, log_sigma = encoder(v1)
    return mu.std((0, 2, 3)) / log_sigma.exp().mean((0, 2, 3)) > threshold
