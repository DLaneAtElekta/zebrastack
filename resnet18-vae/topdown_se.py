# -*- coding: utf-8 -*-
"""Stage-1 top-down routed Squeeze-and-Excitation block.

Drop-in replacement for SqueezeExcitation that accepts an optional top-down
vector `t`. The vector is projected to a per-channel additive bias on the
pre-sigmoid logits of the gate. The projection is zero-initialized so the
module is the identity (== baseline SE) at construction time, letting the
top-down branch learn from zero without perturbing baseline behavior.
"""

from typing import Callable, Optional

import torch
from torch import Tensor, nn


class TopDownRoutedSE(nn.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        td_dim: Optional[int] = None,
        activation: Callable[..., nn.Module] = nn.SiLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()
        if td_dim is not None:
            self.td_proj: Optional[nn.Linear] = nn.Linear(td_dim, input_channels, bias=False)
            # Zero-init so at construction the gate is identical to baseline SE.
            nn.init.zeros_(self.td_proj.weight)
        else:
            self.td_proj = None

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        s = self.avgpool(x)
        s = self.fc1(s)
        s = self.activation(s)
        s = self.fc2(s)
        if t is not None and self.td_proj is not None:
            s = s + self.td_proj(t).view(t.size(0), -1, 1, 1)
        return self.scale_activation(s) * x
