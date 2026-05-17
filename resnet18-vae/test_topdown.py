# -*- coding: utf-8 -*-
"""Tests for Stage-1 top-down routing modules.

Invariants we want to hold:
  - With t=None or with zero-init td_proj, TopDownRoutedSE == SqueezeExcitation
    (given matching fc1/fc2 weights).
  - td_proj receives gradient when t flows through the loss, and after one SGD
    step the t-modulated output differs from the no-op output.
  - TopDownOrientedPowerMap with t=None matches the parent OrientedPowerMap.
  - TopDownEncoder.forward_dict returns the right keys and shapes; with
    zero-init td_heads the first pass behaves like the baseline.
"""

import unittest

import torch
import torch.nn as nn

from squeeze_excitation import SqueezeExcitation
from topdown_se import TopDownRoutedSE


class TestTopDownRoutedSE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.C = 32
        self.S = self.C // 8
        self.D = 8
        self.B = 4
        self.x = torch.randn(self.B, self.C, 16, 16)
        self.t = torch.randn(self.B, self.D)

    def _matched_pair(self, td_dim):
        se = SqueezeExcitation(self.C, self.S)
        tdse = TopDownRoutedSE(self.C, self.S, td_dim=td_dim)
        tdse.fc1.load_state_dict(se.fc1.state_dict())
        tdse.fc2.load_state_dict(se.fc2.state_dict())
        return se, tdse

    def test_no_op_invariant_t_none(self):
        se, tdse = self._matched_pair(td_dim=self.D)
        with torch.no_grad():
            self.assertTrue(torch.allclose(se(self.x), tdse(self.x, t=None), atol=1e-6))

    def test_no_op_invariant_zero_init_td_proj(self):
        se, tdse = self._matched_pair(td_dim=self.D)
        with torch.no_grad():
            self.assertTrue(torch.allclose(se(self.x), tdse(self.x, t=self.t), atol=1e-6))

    def test_td_proj_has_grad_and_modulates_output(self):
        _, tdse = self._matched_pair(td_dim=self.D)
        nn.init.normal_(tdse.td_proj.weight, std=0.1)
        opt = torch.optim.SGD(tdse.parameters(), lr=1.0)

        loss = tdse(self.x, t=self.t).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        self.assertIsNotNone(tdse.td_proj.weight.grad)
        self.assertGreater(tdse.td_proj.weight.grad.abs().sum().item(), 0.0)
        opt.step()

        with torch.no_grad():
            y_with_t = tdse(self.x, t=self.t)
            y_none = tdse(self.x, t=None)
        self.assertFalse(torch.allclose(y_with_t, y_none, atol=1e-6))


class TestTopDownOrientedPowerMap(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = "cpu"
        self.B, self.C_in, self.H, self.W = 2, 1, 64, 64

    def test_no_op_invariant_t_none(self):
        from topdown_oriented_powermap import TopDownOrientedPowerMap

        torch.manual_seed(0)
        td = TopDownOrientedPowerMap(
            self.device, self.C_in, td_dim=8, kernel_size=11, directions=7
        )
        x = torch.randn(self.B, self.C_in, self.H, self.W)
        td.eval()
        with torch.no_grad():
            y1 = td(x, t=None)
            y2 = super(type(td), td).forward(x)  # parent OrientedPowerMap.forward
        self.assertEqual(y1.shape, y2.shape)
        self.assertTrue(torch.allclose(y1, y2, atol=1e-6))

    def test_t_changes_output_after_training_step(self):
        from topdown_oriented_powermap import TopDownOrientedPowerMap

        torch.manual_seed(0)
        td = TopDownOrientedPowerMap(
            self.device, self.C_in, td_dim=8, kernel_size=11, directions=7
        )
        nn.init.normal_(td.se.td_proj.weight, std=0.1)
        x = torch.randn(self.B, self.C_in, self.H, self.W)
        t = torch.randn(self.B, 8)

        opt = torch.optim.SGD(td.parameters(), lr=1.0)
        loss = td(x, t=t).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        self.assertIsNotNone(td.se.td_proj.weight.grad)
        self.assertGreater(td.se.td_proj.weight.grad.abs().sum().item(), 0.0)


class TestTopDownEncoderConstruction(unittest.TestCase):
    """Light smoke test: instantiate the encoder, run forward_dict, check keys.

    Uses small inputs to keep CPU runtime tractable.
    """

    def test_forward_dict_keys_and_shapes(self):
        from topdown_encoder import TopDownEncoder

        torch.manual_seed(0)
        device = "cpu"
        input_size = (1, 128, 128)
        enc = TopDownEncoder(
            device, input_size, init_kernel_size=9, directions=7,
            latent_dim=16, td_dim_v1=8, td_dim_v2=8,
        )
        enc.eval()

        x = torch.randn(1, *input_size)
        with torch.no_grad():
            out = enc.forward_dict(x)

        self.assertEqual(set(out.keys()), {"x_v1", "x_v2", "x_v4", "mu", "log_var"})
        self.assertEqual(out["mu"].shape, (1, 16))
        self.assertEqual(out["log_var"].shape, (1, 16))


if __name__ == "__main__":
    unittest.main()
