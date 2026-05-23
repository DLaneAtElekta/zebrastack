# -*- coding: utf-8 -*-
"""Per-channel Bayesian Gaussian product as a routing layer.

Replaces the multiplicative SE gate with a closed-form posterior over channel
activations. Diagonal precision, single level — the minimal version of the
"vision-as-Brimstone" design.

Math (per spatial location, broadcast over H, W):

    Prior:           p(a)     = N(mu(t),  diag(lambda(t))^{-1})
    Likelihood:      p(e | a) = N(a,      diag(lambda_data)^{-1})
    Posterior mode:  a*       = (lambda * mu + lambda_data * e)
                              / (lambda    + lambda_data)

Initialization is chosen so the layer is (numerically) the identity on `e` at
construction:

  * `to_mu` zero-init -> prior mean is 0.
  * `to_log_diag` zero-weight, large-negative-bias -> prior precision ~= 0.
  * `log_lambda_data` zero-init -> data precision = 1.

So at step 0 the posterior is `(eps * 0 + 1 * e) / (eps + 1) ~= e`. The
top-down pathway therefore starts from a known-good baseline (evidence passes
through) and is learned from zero.

`t=None` is a hard short-circuit that returns `evidence` unchanged, making the
layer a drop-in identity when the encoder is in its bottom-up-only pass.
"""

from typing import Optional

import torch
from torch import Tensor, nn


class BayesianRoutingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        td_dim: Optional[int],
        log_prior_precision_bias_init: float = -10.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.td_dim = td_dim
        # Per-channel data precision; learnable but init'd so likelihood = N(a, I).
        self.log_lambda_data = nn.Parameter(torch.zeros(channels))

        if td_dim is None:
            self.to_mu = None
            self.to_log_diag = None
            return

        self.to_mu = nn.Linear(td_dim, channels)
        self.to_log_diag = nn.Linear(td_dim, channels)
        nn.init.zeros_(self.to_mu.weight)
        nn.init.zeros_(self.to_mu.bias)
        nn.init.zeros_(self.to_log_diag.weight)
        # Large negative bias -> exp(.) ~= 0 -> prior carries no weight at init.
        nn.init.constant_(self.to_log_diag.bias, log_prior_precision_bias_init)

    def forward(self, evidence: Tensor, t: Optional[Tensor] = None) -> Tensor:
        if t is None or self.to_mu is None:
            return evidence
        mu = self.to_mu(t).unsqueeze(-1).unsqueeze(-1)
        lam = self.to_log_diag(t).exp().unsqueeze(-1).unsqueeze(-1)
        lam_d = self.log_lambda_data.exp().view(1, -1, 1, 1)
        return (lam * mu + lam_d * evidence) / (lam + lam_d)
