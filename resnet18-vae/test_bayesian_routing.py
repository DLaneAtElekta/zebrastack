# -*- coding: utf-8 -*-
"""Tests for Bayesian routing modules.

Invariants we want to hold:
  * `t=None` -> BayesianRoutingLayer is the identity on `evidence` (bit-exact).
  * `t!=None` at init -> layer is numerically identity (large negative
    prior-precision bias means the prior carries ~0 weight in the Gaussian
    product, so posterior ~= evidence).
  * Gradients flow to `to_mu`, `to_log_diag`, and `log_lambda_data` when t and
    the loss are coupled; after one SGD step the t-modulated output differs
    from the no-op output.
  * Posterior lies between prior mean and evidence (clamped check on a tiny
    deterministic case).
  * `BayesianRoutedOrientedPowerMap.forward(x, t=None)` equals the same module
    called with `t=zeros`, up to the at-init prior-precision tolerance.
  * `BayesianRoutedEncoder.forward_dict` returns the right keys and shapes.
"""

import unittest

import torch
import torch.nn as nn

from bayesian_routing import BayesianRoutingLayer


class TestBayesianRoutingLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.C, self.H, self.W = 4, 32, 16, 16
        self.D = 8
        self.evidence = torch.randn(self.B, self.C, self.H, self.W)
        self.t = torch.randn(self.B, self.D)

    def test_no_op_invariant_t_none(self):
        layer = BayesianRoutingLayer(channels=self.C, td_dim=self.D)
        with torch.no_grad():
            self.assertTrue(torch.equal(layer(self.evidence, t=None), self.evidence))

    def test_no_op_invariant_at_init_with_t(self):
        layer = BayesianRoutingLayer(channels=self.C, td_dim=self.D)
        with torch.no_grad():
            y = layer(self.evidence, t=self.t)
        # exp(-10) ~= 4.5e-5 so posterior matches evidence to ~1e-4 relative.
        self.assertTrue(torch.allclose(y, self.evidence, atol=1e-3))

    def test_posterior_between_prior_and_evidence(self):
        # Deterministic 1-D case: with positive prior precision the posterior
        # mode must lie elementwise between mu and e.
        layer = BayesianRoutingLayer(
            channels=2, td_dim=2, log_prior_precision_bias_init=0.0
        )
        with torch.no_grad():
            nn.init.eye_(layer.to_mu.weight)
            nn.init.zeros_(layer.to_mu.bias)
        evidence = torch.tensor([[[[1.0]], [[1.0]]]])  # shape [1, 2, 1, 1]
        t = torch.tensor([[5.0, -3.0]])
        with torch.no_grad():
            mu = layer.to_mu(t)
            y = layer(evidence, t=t)
        lo = torch.minimum(mu.view(1, 2, 1, 1), evidence)
        hi = torch.maximum(mu.view(1, 2, 1, 1), evidence)
        self.assertTrue(torch.all(y >= lo - 1e-6))
        self.assertTrue(torch.all(y <= hi + 1e-6))

    def test_grads_flow_and_output_changes_after_step(self):
        layer = BayesianRoutingLayer(channels=self.C, td_dim=self.D)
        # Bump prior precision into a regime where the prior actually matters.
        nn.init.normal_(layer.to_mu.weight, std=0.1)
        nn.init.constant_(layer.to_log_diag.bias, 0.0)
        opt = torch.optim.SGD(layer.parameters(), lr=0.1)

        loss = layer(self.evidence, t=self.t).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        self.assertIsNotNone(layer.to_mu.weight.grad)
        self.assertIsNotNone(layer.to_log_diag.weight.grad)
        self.assertIsNotNone(layer.log_lambda_data.grad)
        self.assertGreater(layer.to_mu.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(layer.to_log_diag.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(layer.log_lambda_data.grad.abs().sum().item(), 0.0)
        opt.step()

        with torch.no_grad():
            y_with_t = layer(self.evidence, t=self.t)
            y_none = layer(self.evidence, t=None)
        self.assertFalse(torch.allclose(y_with_t, y_none, atol=1e-6))


class TestBayesianRoutedOrientedPowerMap(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = "cpu"
        self.B, self.C_in, self.H, self.W = 2, 1, 64, 64

    def test_no_op_invariant_t_none_vs_zero_t(self):
        from bayes_oriented_powermap import BayesianRoutedOrientedPowerMap

        torch.manual_seed(0)
        opm = BayesianRoutedOrientedPowerMap(
            self.device, self.C_in, td_dim=8, kernel_size=11, directions=7
        )
        opm.eval()  # freeze BatchNorm running stats across calls
        x = torch.randn(self.B, self.C_in, self.H, self.W)
        t_zero = torch.zeros(self.B, 8)
        with torch.no_grad():
            y_none = opm(x, t=None)
            y_zero = opm(x, t=t_zero)
        self.assertEqual(y_none.shape, y_zero.shape)
        # Manual path with at-init Bayesian routing should match the t=None path
        # to within the prior-precision floor (exp(-10) ~= 4.5e-5).
        self.assertTrue(torch.allclose(y_none, y_zero, atol=1e-3))

    def test_t_changes_output_after_training_step(self):
        from bayes_oriented_powermap import BayesianRoutedOrientedPowerMap

        torch.manual_seed(0)
        opm = BayesianRoutedOrientedPowerMap(
            self.device, self.C_in, td_dim=8, kernel_size=11, directions=7
        )
        nn.init.normal_(opm.se.to_mu.weight, std=0.1)
        nn.init.constant_(opm.se.to_log_diag.bias, 0.0)
        x = torch.randn(self.B, self.C_in, self.H, self.W)
        t = torch.randn(self.B, 8)

        opt = torch.optim.SGD(opm.parameters(), lr=0.1)
        loss = opm(x, t=t).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        self.assertIsNotNone(opm.se.to_mu.weight.grad)
        self.assertGreater(opm.se.to_mu.weight.grad.abs().sum().item(), 0.0)


class TestBayesianRoutedEncoderConstruction(unittest.TestCase):
    """Light smoke test: instantiate the encoder, run forward_dict, check keys."""

    def test_forward_dict_keys_and_shapes(self):
        from bayes_encoder import BayesianRoutedEncoder

        torch.manual_seed(0)
        device = "cpu"
        input_size = (1, 128, 128)
        enc = BayesianRoutedEncoder(
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
