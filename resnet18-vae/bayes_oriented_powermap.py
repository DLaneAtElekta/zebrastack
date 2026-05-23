# -*- coding: utf-8 -*-
"""OrientedPowerMap that replaces SE with a BayesianRoutingLayer.

Parallel to `TopDownOrientedPowerMap`, but the routing semantics differ:

  * `TopDownOrientedPowerMap` *augments* the SE gate with an additive top-down
    bias on its logits (multiplicative routing).
  * `BayesianRoutedOrientedPowerMap` *replaces* the SE gate with a closed-form
    Gaussian product (Bayesian routing). When `t=None` the routing layer is
    the identity, so the parent's `nn.Sequential` produces "OPM with SE
    swapped for Identity" rather than the original SE-gated baseline.

Channel counts and spatial shapes are unchanged.
"""

from typing import Optional

from torch import Tensor

from bayesian_routing import BayesianRoutingLayer
from oriented_powermap import OrientedPowerMap


class BayesianRoutedOrientedPowerMap(OrientedPowerMap):
    SE_INDEX_IN_CONV = 5  # conv_pre, conv_1, bn, relu, change_res, [SE], conv_2, relu

    def __init__(self, device, in_channels: int, td_dim: Optional[int] = None, **kwargs):
        super().__init__(device, in_channels, **kwargs)
        if td_dim is None:
            self._td_enabled = False
            return
        self._td_enabled = True

        channels = self.se.fc1.in_channels
        new_layer = BayesianRoutingLayer(channels=channels, td_dim=td_dim).to(device)
        self.se = new_layer
        self.conv[self.SE_INDEX_IN_CONV] = new_layer

        self._conv_pre = self.conv[0]
        self._conv_1 = self.conv[1]
        self._bn = self.conv[2]
        self._relu1 = self.conv[3]
        self._change_res = self.conv[4]
        self._conv_2 = self.conv[6]
        self._relu2 = self.conv[7]

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        if t is None or not self._td_enabled:
            # Sequential calls BayesianRoutingLayer with no t -> identity in the SE slot.
            return super().forward(x)

        shortcut_x = self.shortcut(x)
        h = self._conv_pre(x)
        h = self._conv_1(h)
        h = self._bn(h)
        h = self._relu1(h)
        h = self._change_res(h)
        h = self.se(h, t)
        h = self._conv_2(h)
        h = self._relu2(h)
        return 0.5 * (h + shortcut_x)
