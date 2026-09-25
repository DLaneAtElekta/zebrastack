"""Section 7, step FE-3: precision conditioned on context, acting at inference.

With a purely amortized encoder, precision only shapes learning; at test time
mu = encoder(x) ignores it. So context precision needs inference that reads
F: ``settle`` refines the amortized guess by gradient descent on F (a
minimal version of FE-4's settling steps).

Context precision log pi(c) = log pi_base + Delta(c), per V1 channel:
  * ``context_precision_fe``: Delta(c) chosen by minimizing F on scenes
    from context c. For a precision vector the optimum is closed-form (1 /
    residual variance after settling), iterated with settling.
  * ``expected_signal_precision``: Delta(c) = log(1 + expected target signal
    variance / residual variance) per channel, i.e. precision raised where the
    context predicts the target adds signal (attention as the precision of
    expected signals; Feldman & Friston 2010). A comparator, not an F optimum.
"""

import torch

from .decoders import ConvFactorAnalysis
from .free_energy import AmortizedEncoder, TICAPrior


def accuracy_curvature(decoder: ConvFactorAnalysis, log_precision: torch.Tensor, shape: tuple,
                       n_iter: int = 20, border: int = 2, seed: int = 0) -> float:
    """Largest eigenvalue of G^T Pi G (the accuracy term's Hessian in mu) by
    power iteration; gradient steps below 2 / L on the accuracy term are stable."""
    lp = log_precision.view(1, -1, 1, 1) if log_precision.dim() else log_precision
    pi = lp.exp()
    v = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    v = v / v.norm()
    zero = decoder.mean(torch.zeros(shape)).detach()
    lam = 0.0
    with torch.enable_grad():
        for _ in range(n_iter):
            v.requires_grad_(True)
            gv = (decoder.mean(v) - zero)[..., border:-border, border:-border]
            (hv,) = torch.autograd.grad((0.5 * pi * gv.pow(2)).sum(), v)
            lam = float((hv * v).sum().detach())
            v = (hv / hv.norm()).detach()
    return lam


def settle(
    x: torch.Tensor,
    mu0: torch.Tensor,
    decoder: ConvFactorAnalysis,
    prior: TICAPrior,
    log_precision: torch.Tensor,
    n_steps: int = 20,
    step: float = 0.5,
    border: int = 2,
) -> torch.Tensor:
    """Refine posterior means by gradient descent on
    F(mu) = sum 1/2 pi (x - g(mu))^2 + TICA prior(mu)  (entropy is constant in mu).

    The step is ``step`` / L, with L the accuracy term's curvature (power
    iteration), so the same ``step`` is stable for any precision or decoder."""
    lp = log_precision.view(1, -1, 1, 1) if log_precision.dim() else log_precision
    pi = lp.exp()
    eta = step / accuracy_curvature(decoder, log_precision, (1, *mu0.shape[1:]), border=border)
    mu = mu0.detach().clone()
    with torch.enable_grad():
        for _ in range(n_steps):
            mu.requires_grad_(True)
            err = (x - decoder.mean(mu))[..., border:-border, border:-border]
            f = (0.5 * pi * err.pow(2)).sum() + prior.neg_log(mu).sum()
            (grad,) = torch.autograd.grad(f, mu)
            mu = (mu - eta * grad).detach()
    return mu


def settle_lbfgs(
    x: torch.Tensor,
    mu0: torch.Tensor,
    decoder: ConvFactorAnalysis,
    prior: TICAPrior,
    log_precision: torch.Tensor,
    n_steps: int = 100,
    border: int = 2,
) -> torch.Tensor:
    """Settle to (near) convergence with L-BFGS: the accuracy term is stiff, so
    plain gradient steps barely move the low-curvature directions."""
    lp = log_precision.view(1, -1, 1, 1) if log_precision.dim() else log_precision
    pi = lp.exp().detach()
    mu = mu0.detach().clone(memory_format=torch.contiguous_format).requires_grad_(True)
    opt = torch.optim.LBFGS([mu], max_iter=n_steps, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        err = (x - decoder.mean(mu))[..., border:-border, border:-border]
        f = (0.5 * pi * err.pow(2)).sum() + prior.neg_log(mu).sum()
        f.backward()
        return f

    with torch.enable_grad():
        opt.step(closure)
    return mu.detach()


def residual_variance(x: torch.Tensor, mu: torch.Tensor, decoder: ConvFactorAnalysis, border: int = 2) -> torch.Tensor:
    with torch.no_grad():
        err = (x - decoder.mean(mu))[..., border:-border, border:-border]
        return err.pow(2).mean((0, 2, 3))


def context_precision_fe(
    v1: torch.Tensor,
    encoder: AmortizedEncoder,
    decoder: ConvFactorAnalysis,
    prior: TICAPrior,
    base_log_precision: torch.Tensor,
    n_rounds: int = 3,
    **settle_kw,
) -> torch.Tensor:
    """F-optimal per-channel log precision for scenes ``v1`` from one context:
    alternate settling and the closed-form update log pi = -log residual variance."""
    x = decoder.from_v1(v1)
    with torch.no_grad():
        mu0, _ = encoder(v1)
    lp = base_log_precision.clone()
    for _ in range(n_rounds):
        mu = settle(x, mu0, decoder, prior, lp, **settle_kw)
        lp = -residual_variance(x, mu, decoder).log()
    return lp


def expected_signal_precision(
    v1_target: torch.Tensor,
    v1_absent: torch.Tensor,
    decoder: ConvFactorAnalysis,
    base_log_precision: torch.Tensor,
    window: slice,
) -> torch.Tensor:
    """log pi_base + log(1 + signal / noise) per channel, where signal is the
    extra within-window variance the target adds (target minus absent scenes)
    and noise is the base residual variance 1 / pi_base."""
    xt = decoder.from_v1(v1_target)[..., window, window]
    xa = decoder.from_v1(v1_absent)[..., window, window]
    signal = (xt.var((2, 3)).mean(0) - xa.var((2, 3)).mean(0)).clamp(min=0)
    noise = (-base_log_precision).exp()
    return base_log_precision + torch.log1p(signal / noise)
