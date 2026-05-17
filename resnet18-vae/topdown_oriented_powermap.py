# -*- coding: utf-8 -*-
"""OrientedPowerMap wrapper whose SE block accepts a top-down vector.

The parent OrientedPowerMap bakes its SE block into an nn.Sequential, which
cannot pass extra arguments through. This subclass:
  1. Lets the parent construct everything normally.
  2. Swaps the SqueezeExcitation for a TopDownRoutedSE, copying fc weights so
     behavior with t=None matches the parent at init.
  3. Overrides forward to run the chain manually when t is supplied.

Channel counts and spatial shapes are unchanged, so this is a drop-in.
"""

from typing import Optional

import torch
from torch import Tensor

from oriented_powermap import OrientedPowerMap
from topdown_se import TopDownRoutedSE


class TopDownOrientedPowerMap(OrientedPowerMap):
    SE_INDEX_IN_CONV = 5  # conv_pre, conv_1, bn, relu, change_res, [SE], conv_2, relu

    def __init__(self, device, in_channels: int, td_dim: Optional[int] = None, **kwargs):
        super().__init__(device, in_channels, **kwargs)
        if td_dim is None:
            self._td_enabled = False
            return
        self._td_enabled = True

        old_se = self.se
        new_se = TopDownRoutedSE(
            input_channels=old_se.fc1.in_channels,
            squeeze_channels=old_se.fc1.out_channels,
            td_dim=td_dim,
        )
        new_se.fc1.load_state_dict(old_se.fc1.state_dict())
        new_se.fc2.load_state_dict(old_se.fc2.state_dict())
        new_se = new_se.to(device)

        self.se = new_se
        self.conv[self.SE_INDEX_IN_CONV] = new_se

        self._conv_pre = self.conv[0]
        self._conv_1 = self.conv[1]
        self._bn = self.conv[2]
        self._relu1 = self.conv[3]
        self._change_res = self.conv[4]
        self._conv_2 = self.conv[6]
        self._relu2 = self.conv[7]

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        if t is None or not self._td_enabled:
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
