"""
Unit tests for variational Bayesian feature visualization.

Tests each component independently:
1. Reparameterization trick
2. Prior distributions (Laplace, sparse Gaussian, kurtosis)
3. Hessian entropy estimation
4. End-to-end variational visualizer
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from variational_visualization import (
    VariationalImageParameter,
    ActivationDistributionPrior,
    HessianEntropyEstimator,
    VariationalFeatureVisualizer
)


class TestVariationalImageParameter:
    """Test reparameterization trick implementation."""

    def test_initialization(self):
        """Test that parameters are initialized correctly."""
        param = VariationalImageParameter(image_size=64, channels=3)

        assert param.mu.shape == (1, 3, 64, 64)
        assert param.log_sigma.shape == (1, 3, 64, 64)
        assert param.mu.requires_grad
        assert param.log_sigma.requires_grad

    def test_sigma_property(self):
        """Test that sigma is computed correctly from log_sigma."""
        param = VariationalImageParameter(image_size=32, channels=3)

        # Set log_sigma to known value
        param.log_sigma.data = torch.ones_like(param.log_sigma) * 2.0

        # sigma = exp(0.5 * log_sigma) = exp(1.0) ≈ 2.718
        expected_sigma = np.exp(1.0)
        actual_sigma = param.sigma

        assert torch.allclose(actual_sigma, torch.tensor(expected_sigma, dtype=torch.float32), atol=1e-5)

    def test_sampling_shape(self):
        """Test that sampling produces correct shape."""
        param = VariationalImageParameter(image_size=32, channels=3)

        samples = param.sample(n_samples=10)

        assert samples.shape == (10, 3, 32, 32)

    def test_sampling_mean(self):
        """Test that sample mean converges to mu."""
        param = VariationalImageParameter(image_size=16, channels=3)

        # Set mu to known value
        param.mu.data = torch.ones_like(param.mu) * 5.0

        # Set sigma to small value for low variance
        param.log_sigma.data = torch.ones_like(param.log_sigma) * -6.0

        # Sample many times
        samples = param.sample(n_samples=1000)

        # Mean should be close to mu
        sample_mean = samples.mean(dim=0, keepdim=True)

        assert torch.allclose(sample_mean, param.mu, atol=0.1)

    def test_sampling_variance(self):
        """Test that sample variance matches sigma^2."""
        param = VariationalImageParameter(image_size=16, channels=3)

        # Set mu to zero
        param.mu.data = torch.zeros_like(param.mu)

        # Set sigma to known value
        param.log_sigma.data = torch.ones_like(param.log_sigma) * 0.0  # sigma = 1.0

        # Sample many times
        samples = param.sample(n_samples=10000)

        # Variance should be close to sigma^2 = 1.0
        sample_var = samples.var(dim=0, keepdim=True)
        expected_var = param.get_variance()

        assert torch.allclose(sample_var, expected_var, atol=0.1)

    def test_temperature_scaling(self):
        """Test that temperature scales variance."""
        param = VariationalImageParameter(image_size=16, channels=3)

        # Sample with different temperatures
        samples_t1 = param.sample(n_samples=1000, temperature=1.0)
        samples_t2 = param.sample(n_samples=1000, temperature=2.0)

        var_t1 = samples_t1.var()
        var_t2 = samples_t2.var()

        # Variance should scale with temperature^2
        # var(t=2) ≈ 4 * var(t=1)
        ratio = var_t2 / var_t1
        assert torch.allclose(ratio, torch.tensor(4.0), atol=0.5)

    def test_kl_divergence_standard_normal(self):
        """Test KL divergence to standard normal."""
        param = VariationalImageParameter(image_size=8, channels=3)

        # Case 1: q = N(0, 1) should have KL = 0
        param.mu.data = torch.zeros_like(param.mu)
        param.log_sigma.data = torch.zeros_like(param.log_sigma)

        kl = param.kl_divergence_standard_normal()
        assert torch.allclose(kl, torch.tensor(0.0), atol=1e-4)

        # Case 2: q = N(1, 1) should have KL > 0
        param.mu.data = torch.ones_like(param.mu)
        param.log_sigma.data = torch.zeros_like(param.log_sigma)

        kl = param.kl_divergence_standard_normal()
        assert kl > 0

    def test_gradient_flow(self):
        """Test that gradients flow through reparameterization."""
        param = VariationalImageParameter(image_size=8, channels=3)

        # Sample and compute loss
        samples = param.sample(n_samples=5)
        loss = samples.mean()

        # Backprop
        loss.backward()

        # Both mu and log_sigma should have gradients
        assert param.mu.grad is not None
        assert param.log_sigma.grad is not None
        assert not torch.allclose(param.mu.grad, torch.zeros_like(param.mu.grad))


class TestActivationDistributionPrior:
    """Test prior distribution implementations."""

    def test_laplace_prior_shape(self):
        """Test that Laplace prior returns scalar."""
        prior = ActivationDistributionPrior()

        activations = torch.randn(10, 7, 7)
        loss = prior.laplace_prior(activations)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0  # NLL should be non-negative

    def test_laplace_prior_sparsity(self):
        """Test that Laplace prior encourages sparsity."""
        prior = ActivationDistributionPrior()

        # Sparse activations (mostly zeros)
        sparse = torch.zeros(100)
        sparse[0] = 10.0

        # Dense activations (all non-zero)
        dense = torch.ones(100) * 2.0

        sparse_loss = prior.laplace_prior(sparse, target_mean=0.0, target_scale=1.0)
        dense_loss = prior.laplace_prior(dense, target_mean=0.0, target_scale=1.0)

        # Sparse should have lower loss for Laplace prior
        assert sparse_loss < dense_loss

    def test_sparse_gaussian_prior_shape(self):
        """Test sparse Gaussian prior returns scalar."""
        prior = ActivationDistributionPrior()

        activations = torch.randn(10, 7, 7)
        loss = prior.sparse_gaussian_prior(activations)

        assert loss.shape == torch.Size([])

    def test_sparse_gaussian_prior_target(self):
        """Test that sparse Gaussian prefers target mean."""
        prior = ActivationDistributionPrior()

        target_mean = 5.0

        # Activations at target
        at_target = torch.ones(100) * target_mean

        # Activations away from target
        away = torch.ones(100) * 0.0

        loss_at = prior.sparse_gaussian_prior(at_target, target_mean=target_mean)
        loss_away = prior.sparse_gaussian_prior(away, target_mean=target_mean)

        assert loss_at < loss_away

    def test_kurtosis_matching_loss_normal(self):
        """Test kurtosis matching for normal distribution."""
        prior = ActivationDistributionPrior()

        # Generate normal samples (should have kurtosis ≈ 3)
        torch.manual_seed(42)
        normal_samples = torch.randn(10000)

        loss = prior.kurtosis_matching_loss(normal_samples, target_kurtosis=3.0)

        # Loss should be small for normal distribution
        assert loss < 1.0

    def test_kurtosis_matching_loss_uniform(self):
        """Test kurtosis matching for uniform distribution."""
        prior = ActivationDistributionPrior()

        # Uniform distribution has kurtosis ≈ 1.8
        uniform_samples = torch.rand(10000) * 2 - 1

        loss_3 = prior.kurtosis_matching_loss(uniform_samples, target_kurtosis=3.0)
        loss_18 = prior.kurtosis_matching_loss(uniform_samples, target_kurtosis=1.8)

        # Should be closer to 1.8 than 3.0
        assert loss_18 < loss_3

    def test_kurtosis_matching_loss_heavy_tail(self):
        """Test kurtosis matching for heavy-tailed distribution."""
        prior = ActivationDistributionPrior()

        # Laplace has kurtosis = 6
        laplace_samples = torch.distributions.Laplace(0, 1).sample((10000,))

        loss_3 = prior.kurtosis_matching_loss(laplace_samples, target_kurtosis=3.0)
        loss_6 = prior.kurtosis_matching_loss(laplace_samples, target_kurtosis=6.0)

        # Should be closer to 6.0 than 3.0
        assert loss_6 < loss_3

    def test_gradient_flow_through_priors(self):
        """Test that gradients flow through all priors."""
        prior = ActivationDistributionPrior()

        activations = torch.randn(100, requires_grad=True)

        # Test each prior
        loss_laplace = prior.laplace_prior(activations)
        loss_laplace.backward()
        assert activations.grad is not None

        activations.grad = None
        loss_sparse = prior.sparse_gaussian_prior(activations)
        loss_sparse.backward()
        assert activations.grad is not None

        activations.grad = None
        loss_kurtosis = prior.kurtosis_matching_loss(activations)
        loss_kurtosis.backward()
        assert activations.grad is not None


class TestHessianEntropyEstimator:
    """Test Hessian-based entropy estimation."""

    def test_diagonal_hessian_simple_quadratic(self):
        """Test diagonal Hessian on simple quadratic function."""
        # f(x) = 0.5 * x^T A x where A is diagonal
        # Hessian = A

        x = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        a = torch.tensor([2.0, 4.0, 6.0])

        loss = 0.5 * torch.sum(a * x ** 2)

        entropy = HessianEntropyEstimator.diagonal_hessian_entropy(loss, x)

        # Should return a scalar
        assert entropy.shape == torch.Size([])

    def test_diagonal_hessian_gradient_dependence(self):
        """Test that Hessian entropy changes with curvature."""
        # High curvature (steep)
        x_steep = nn.Parameter(torch.randn(10))
        loss_steep = torch.sum(100 * x_steep ** 2)

        # Low curvature (flat)
        x_flat = nn.Parameter(torch.randn(10))
        loss_flat = torch.sum(0.1 * x_flat ** 2)

        # Entropy should be higher for flatter function
        # (larger basin = higher entropy)
        # But our implementation uses -log(H_ii), so steep has higher entropy
        # This is inverted - we're computing -0.5 log(curvature)
        # Higher curvature = lower entropy (tighter basin)

        entropy_steep = HessianEntropyEstimator.diagonal_hessian_entropy(loss_steep, x_steep)
        entropy_flat = HessianEntropyEstimator.diagonal_hessian_entropy(loss_flat, x_flat)

        # Flat should have higher entropy
        assert entropy_flat > entropy_steep

    def test_hutchinson_trace_estimator_shape(self):
        """Test Hutchinson estimator returns scalar."""
        x = nn.Parameter(torch.randn(5, 5))
        loss = torch.sum(x ** 2)

        trace = HessianEntropyEstimator.hutchinson_trace_estimator(
            loss, x, n_samples=5
        )

        assert trace.shape == torch.Size([])

    def test_hutchinson_trace_estimator_quadratic(self):
        """Test Hutchinson estimator on quadratic with known Hessian."""
        # f(x) = 0.5 * x^T A x
        # Hessian = A (constant)
        # Trace(A) should be close to sum of diagonal

        n = 10
        x = nn.Parameter(torch.randn(n))

        # Diagonal matrix
        a = torch.rand(n) * 2 + 1  # Values between 1 and 3

        loss = 0.5 * torch.sum(a * x ** 2)

        trace_estimate = HessianEntropyEstimator.hutchinson_trace_estimator(
            loss, x, n_samples=100
        )

        # True trace
        true_trace = a.sum()

        # Should be approximately equal
        assert torch.allclose(trace_estimate, true_trace, rtol=0.3)


class TestVariationalFeatureVisualizer:
    """Test end-to-end variational visualizer."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple CNN for testing."""
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                return x

        return SimpleCNN()

    def test_initialization(self, simple_model):
        """Test visualizer initialization."""
        viz = VariationalFeatureVisualizer(
            model=simple_model,
            layer_name='conv1',
            channel_idx=0,
            image_size=64
        )

        assert viz.layer_name == 'conv1'
        assert viz.channel_idx == 0
        assert viz.image_param.image_size == 64

    def test_hook_registration(self, simple_model):
        """Test that hooks can be registered."""
        viz = VariationalFeatureVisualizer(
            model=simple_model,
            layer_name='conv1',
            channel_idx=0
        )

        viz.register_hook()

        # Hook should be registered
        assert viz._hook_handle is not None

        # Test forward pass captures activations
        dummy_input = torch.randn(1, 3, 64, 64)
        _ = simple_model(dummy_input)

        assert viz.activations is not None
        assert viz.activations.shape[1] == 16  # conv1 has 16 channels

        viz.remove_hook()

    def test_forward_pass(self, simple_model):
        """Test forward pass samples and captures activations."""
        viz = VariationalFeatureVisualizer(
            model=simple_model,
            layer_name='conv1',
            channel_idx=5,
            image_size=64
        )

        viz.register_hook()

        images, activations = viz.forward(n_samples=8)

        assert images.shape == (8, 3, 64, 64)
        assert activations.shape == (8, 64, 64)  # Spatial dimensions

        viz.remove_hook()

    def test_compute_loss_laplace(self, simple_model):
        """Test loss computation with Laplace prior."""
        viz = VariationalFeatureVisualizer(
            model=simple_model,
            layer_name='conv1',
            channel_idx=0,
            prior_type='laplace',
            entropy_weight=0.0  # Disable entropy for speed
        )

        viz.register_hook()

        images, activations = viz.forward(n_samples=4)
        losses = viz.compute_loss(activations, compute_entropy=False)

        assert 'kl_loss' in losses
        assert 'total_loss' in losses
        assert losses['kl_loss'].item() >= 0

        viz.remove_hook()

    def test_compute_loss_sparse_gaussian(self, simple_model):
        """Test loss computation with sparse Gaussian prior."""
        viz = VariationalFeatureVisualizer(
            model=simple_model,
            layer_name='conv1',
            channel_idx=0,
            prior_type='sparse_gaussian',
            entropy_weight=0.0
        )

        viz.register_hook()

        images, activations = viz.forward(n_samples=4)
        losses = viz.compute_loss(activations, compute_entropy=False)

        assert 'kl_loss' in losses
        assert 'total_loss' in losses

        viz.remove_hook()

    def test_compute_loss_kurtosis(self, simple_model):
        """Test loss computation with kurtosis prior."""
        viz = VariationalFeatureVisualizer(
            model=simple_model,
            layer_name='conv1',
            channel_idx=0,
            prior_type='kurtosis',
            entropy_weight=0.0
        )

        viz.register_hook()

        images, activations = viz.forward(n_samples=4)
        losses = viz.compute_loss(
            activations,
            compute_entropy=False,
            prior_kwargs={'target_kurtosis': 5.0}
        )

        assert 'kl_loss' in losses

        viz.remove_hook()

    def test_compute_loss_with_entropy(self, simple_model):
        """Test loss computation with entropy regularization."""
        viz = VariationalFeatureVisualizer(
            model=simple_model,
            layer_name='conv1',
            channel_idx=0,
            prior_type='laplace',
            entropy_weight=0.1
        )

        viz.register_hook()

        images, activations = viz.forward(n_samples=4)
        losses = viz.compute_loss(activations, compute_entropy=True)

        assert 'entropy' in losses
        assert 'entropy_loss' in losses
        assert 'total_loss' in losses

        viz.remove_hook()

    def test_optimization_step(self, simple_model):
        """Test that one optimization step updates parameters."""
        viz = VariationalFeatureVisualizer(
            model=simple_model,
            layer_name='conv1',
            channel_idx=0,
            prior_type='sparse_gaussian',
            entropy_weight=0.0
        )

        viz.register_hook()

        # Store initial parameters
        initial_mu = viz.image_param.mu.data.clone()

        # Optimization step
        optimizer = torch.optim.Adam(viz.image_param.parameters(), lr=0.1)

        images, activations = viz.forward(n_samples=4)
        losses = viz.compute_loss(
            activations,
            compute_entropy=False,
            prior_kwargs={'target_mean': 5.0}
        )

        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        # Parameters should have changed
        assert not torch.allclose(viz.image_param.mu, initial_mu)

        viz.remove_hook()

    def test_get_visualization(self, simple_model):
        """Test getting visualization output."""
        viz = VariationalFeatureVisualizer(
            model=simple_model,
            layer_name='conv1',
            channel_idx=0,
            image_size=32
        )

        vis = viz.get_visualization()

        assert vis.shape == (1, 3, 32, 32)
        assert isinstance(vis, torch.Tensor)


def run_all_tests():
    """Run all tests and report results."""
    print("🧪 Running Variational Visualization Tests")
    print("=" * 60)

    # Run pytest programmatically
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes'
    ])


if __name__ == "__main__":
    run_all_tests()
