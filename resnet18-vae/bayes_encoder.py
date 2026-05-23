# -*- coding: utf-8 -*-
"""Encoder with Bayesian top-down routing into V1 and V2.

Parallel to `TopDownEncoder`. Same two-pass structure (pass 1 = bottom-up to
`mu_0`; pass 2 = project `mu_0` into per-stage top-down vectors and rerun the
routed stages), but the routing layer is `BayesianRoutingLayer` rather than
`TopDownRoutedSE`.

Heads are zero-initialized; with the BayesianRoutingLayer's at-init
near-identity behavior the first training step is numerically equivalent to
"baseline Encoder with SE replaced by Identity in V1/V2". Not bit-identical to
the SE-gated baseline, but a known, stable starting point.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from bayes_oriented_powermap import BayesianRoutedOrientedPowerMap
from encoder import Encoder


class BayesianRoutedEncoder(Encoder):
    def __init__(
        self,
        device,
        input_size,
        init_kernel_size: int = 9,
        directions: int = 7,
        latent_dim: int = 32,
        td_dim_v1: int = 64,
        td_dim_v2: int = 64,
    ):
        super().__init__(
            device,
            input_size,
            init_kernel_size=init_kernel_size,
            directions=directions,
            latent_dim=latent_dim,
        )

        self.oriented_powermap = BayesianRoutedOrientedPowerMap(
            device,
            input_size[0],
            kernel_size=init_kernel_size,
            frequencies=None,
            directions=directions,
            out_res="^2",
            use_powermap=False,
            td_dim=td_dim_v1,
        ).to(device)

        self.oriented_powermap_2 = BayesianRoutedOrientedPowerMap(
            device,
            self.oriented_powermap.out_channels,
            kernel_size=init_kernel_size,
            frequencies=None,
            directions=directions,
            out_res="^2",
            td_dim=td_dim_v2,
        ).to(device)

        self.td_head_v1 = nn.Linear(latent_dim, td_dim_v1).to(device)
        self.td_head_v2 = nn.Linear(latent_dim, td_dim_v2).to(device)
        nn.init.zeros_(self.td_head_v1.weight)
        nn.init.zeros_(self.td_head_v1.bias)
        nn.init.zeros_(self.td_head_v2.weight)
        nn.init.zeros_(self.td_head_v2.bias)

    def _from_v2_to_latent(self, x_v2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x_v4 = self.oriented_powermap_3(x_v2)
        h = self.residual_blocks(x_v4)

        h = self.penpenultimate_conv1(h)
        bypass = torch.clone(h)
        for _ in range(3):
            h = self.penpenultimate_residual(h)
        h = 0.5 * (bypass + h)
        h = self.penpenultimate_decimate(h)

        h = self.penultimate_conv1(h)
        bypass = torch.clone(h)
        for _ in range(6):
            h = self.penultimate_residual(h)
        h = 0.5 * (bypass + h)
        h = self.penultimate_decimate(h)

        h = self.final_conv1(h)
        bypass = torch.clone(h)
        for _ in range(3):
            h = self.final_residual(h)
        h = 0.5 * (bypass + h)
        h = self.final_decimate(h)

        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return x_v4, mu, log_var

    def forward_dict(self, x: Tensor) -> dict:
        x_v1 = self.oriented_powermap(x)
        x_v2 = self.oriented_powermap_2(x_v1)
        _, mu_0, _ = self._from_v2_to_latent(x_v2)

        t_v1 = self.td_head_v1(mu_0)
        t_v2 = self.td_head_v2(mu_0)

        x_v1 = self.oriented_powermap(x, t=t_v1)
        x_v2 = self.oriented_powermap_2(x_v1, t=t_v2)
        x_v4, mu, log_var = self._from_v2_to_latent(x_v2)

        return {"x_v1": x_v1, "x_v2": x_v2, "x_v4": x_v4, "mu": mu, "log_var": log_var}

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        result = self.forward_dict(x)
        return result["mu"], result["log_var"]
