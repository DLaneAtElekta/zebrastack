"""
Variational Bayesian Feature Visualization.

This module implements feature visualization as a variational inference problem:
- Input parameterized as a distribution q(x) using reparameterization trick
- Target posterior p(a) with sparsity/kurtosis priors
- Entropy regularization via Hessian estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace, Normal
from typing import Optional, Tuple, Dict, Any
import numpy as np


class VariationalImageParameter(nn.Module):
    """
    Parameterizes the input image as a Gaussian distribution using
    the reparameterization trick for gradient-based optimization.

    q(x) = N(μ, diag(σ²))

    Samples are generated as: x = μ + σ ⊙ ε, where ε ~ N(0, I)
    """

    def __init__(
        self,
        image_size: int = 224,
        channels: int = 3,
        initial_mu: Optional[torch.Tensor] = None,
        initial_log_sigma: float = -2.0,
        device: str = 'cpu'
    ):
        """
        Initialize variational image parameters.

        Args:
            image_size: Height/width of square image
            channels: Number of color channels
            initial_mu: Initial mean (random if None)
            initial_log_sigma: Initial log variance (scalar)
            device: Device to place parameters on
        """
        super().__init__()

        self.image_size = image_size
        self.channels = channels
        self.device = device

        # Mean of the distribution
        if initial_mu is None:
            initial_mu = torch.randn(1, channels, image_size, image_size) * 0.01
        self.mu = nn.Parameter(initial_mu.to(device))

        # Log variance (for numerical stability)
        self.log_sigma = nn.Parameter(
            torch.ones(1, channels, image_size, image_size).to(device) * initial_log_sigma
        )

    @property
    def sigma(self) -> torch.Tensor:
        """Get standard deviation (σ = exp(0.5 * log_sigma))."""
        return torch.exp(0.5 * self.log_sigma)

    def sample(self, n_samples: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample images using reparameterization trick.

        Args:
            n_samples: Number of samples to draw
            temperature: Temperature parameter (scales variance)

        Returns:
            Tensor of shape (n_samples, channels, height, width)
        """
        # Sample noise
        eps = torch.randn(n_samples, *self.mu.shape[1:], device=self.device)

        # Reparameterization: x = μ + temperature * σ * ε
        samples = self.mu + temperature * self.sigma * eps

        return samples

    def get_mean(self) -> torch.Tensor:
        """Get the mean image (mode of the distribution)."""
        return self.mu.clone()

    def get_variance(self) -> torch.Tensor:
        """Get the variance."""
        return torch.exp(self.log_sigma)

    def kl_divergence_standard_normal(self) -> torch.Tensor:
        """
        Compute KL divergence between q(x) and standard normal N(0, I).

        KL(q || N(0,I)) = 0.5 * sum(μ² + σ² - log(σ²) - 1)

        Returns:
            Scalar KL divergence
        """
        mu_sq = self.mu.pow(2)
        sigma_sq = torch.exp(self.log_sigma)

        kl = 0.5 * torch.sum(mu_sq + sigma_sq - self.log_sigma - 1)
        return kl


class ActivationDistributionPrior:
    """
    Defines prior distributions over activations with sparsity and kurtosis control.
    """

    @staticmethod
    def laplace_prior(
        activations: torch.Tensor,
        target_mean: float = 0.0,
        target_scale: float = 1.0,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Negative log-likelihood under Laplace distribution (encourages sparsity).

        Args:
            activations: Activation values
            target_mean: Location parameter (μ)
            target_scale: Scale parameter (b), controls sparsity
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Negative log-likelihood
        """
        dist = Laplace(target_mean, target_scale)
        nll = -dist.log_prob(activations)

        if reduction == 'mean':
            return nll.mean()
        elif reduction == 'sum':
            return nll.sum()
        else:
            return nll

    @staticmethod
    def sparse_gaussian_prior(
        activations: torch.Tensor,
        target_mean: float = 5.0,  # High activation target
        target_std: float = 1.0,
        sparsity_weight: float = 0.1,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Gaussian prior with L1 regularization for sparsity.

        Args:
            activations: Activation values
            target_mean: Target mean activation
            target_std: Target standard deviation
            sparsity_weight: Weight for L1 sparsity term
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Negative log-likelihood with sparsity penalty
        """
        # Gaussian NLL
        dist = Normal(target_mean, target_std)
        nll = -dist.log_prob(activations)

        # L1 sparsity penalty
        l1_penalty = sparsity_weight * torch.abs(activations - target_mean)

        loss = nll + l1_penalty

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    @staticmethod
    def kurtosis_matching_loss(
        activations: torch.Tensor,
        target_kurtosis: float = 3.0,  # 3.0 = normal distribution
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Loss that encourages activations to match a target kurtosis.

        Kurtosis = E[(x - μ)^4] / (E[(x - μ)^2])^2
        - Kurtosis = 3: Normal distribution
        - Kurtosis > 3: Heavy tails, peaky (super-Gaussian)
        - Kurtosis < 3: Light tails, flat (sub-Gaussian)

        Args:
            activations: Activation values
            target_kurtosis: Target kurtosis value
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Squared difference from target kurtosis
        """
        # Flatten to compute statistics
        flat = activations.flatten()

        # Compute moments
        mean = flat.mean()
        centered = flat - mean

        # Variance (2nd moment)
        var = (centered ** 2).mean()

        # 4th moment
        fourth_moment = (centered ** 4).mean()

        # Kurtosis (with small epsilon for numerical stability)
        kurtosis = fourth_moment / (var ** 2 + 1e-8)

        # Loss is squared difference from target
        loss = (kurtosis - target_kurtosis) ** 2

        return loss


class HessianEntropyEstimator:
    """
    Estimates entropy of the activation distribution using Hessian-based approximations.

    For a Gaussian approximation: H ≈ 0.5 * log(det(H^{-1}))
    where H is the Hessian of the loss w.r.t. parameters.
    """

    @staticmethod
    def diagonal_hessian_entropy(
        loss: torch.Tensor,
        parameters: torch.Tensor,
        create_graph: bool = True
    ) -> torch.Tensor:
        """
        Estimate entropy using diagonal Hessian approximation.

        H ≈ 0.5 * sum(log(1 / H_ii)) = -0.5 * sum(log(H_ii))

        Args:
            loss: Scalar loss value
            parameters: Parameters to compute Hessian w.r.t.
            create_graph: Whether to create computation graph

        Returns:
            Entropy estimate (scalar)
        """
        # Flatten parameters
        params_flat = parameters.flatten()

        # First derivative
        grads = torch.autograd.grad(
            loss,
            parameters,
            create_graph=create_graph,
            retain_graph=True
        )[0]

        # Compute diagonal elements of Hessian
        # For each gradient element, take derivative again
        hessian_diag = []
        grads_flat = grads.flatten()

        for i in range(min(len(grads_flat), 1000)):  # Limit for efficiency
            if create_graph:
                grad2 = torch.autograd.grad(
                    grads_flat[i],
                    parameters,
                    retain_graph=True,
                    create_graph=False
                )[0]
                hessian_diag.append(grad2.flatten()[i])

        # Convert to tensor
        if hessian_diag:
            h_diag = torch.stack(hessian_diag)

            # Add small constant for numerical stability
            h_diag = torch.abs(h_diag) + 1e-6

            # Entropy estimate: -0.5 * sum(log(H_ii))
            entropy = -0.5 * torch.sum(torch.log(h_diag))

            return entropy
        else:
            return torch.tensor(0.0, device=parameters.device)

    @staticmethod
    def hutchinson_trace_estimator(
        loss: torch.Tensor,
        parameters: torch.Tensor,
        n_samples: int = 10
    ) -> torch.Tensor:
        """
        Estimate trace of Hessian using Hutchinson's stochastic estimator.

        Tr(H) ≈ E[v^T H v] where v ~ N(0, I)

        Args:
            loss: Scalar loss value
            parameters: Parameters to compute Hessian w.r.t.
            n_samples: Number of random vectors for estimation

        Returns:
            Trace estimate (scalar)
        """
        params_flat = parameters.flatten()
        n_params = len(params_flat)

        # First gradient
        grads = torch.autograd.grad(
            loss,
            parameters,
            create_graph=True,
            retain_graph=True
        )[0]

        trace_estimates = []

        for _ in range(n_samples):
            # Random Rademacher vector (±1)
            v = torch.randint(0, 2, (n_params,), device=parameters.device).float() * 2 - 1

            # Compute Hessian-vector product: H @ v
            grads_flat = grads.flatten()
            hvp = torch.autograd.grad(
                grads_flat,
                parameters,
                grad_outputs=v,
                retain_graph=True,
                create_graph=False
            )[0]

            # v^T (H @ v)
            trace_estimate = torch.dot(v, hvp.flatten())
            trace_estimates.append(trace_estimate)

        # Average over samples
        trace = torch.stack(trace_estimates).mean()

        return trace


class VariationalFeatureVisualizer(nn.Module):
    """
    Main class for variational Bayesian feature visualization.

    Optimizes the input distribution to match a target posterior over activations
    with entropy regularization.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_name: str,
        channel_idx: int,
        image_size: int = 224,
        prior_type: str = 'laplace',
        entropy_weight: float = 0.01,
        device: str = 'cpu'
    ):
        """
        Initialize variational visualizer.

        Args:
            model: Neural network model
            layer_name: Target layer name
            channel_idx: Target channel index
            image_size: Image size
            prior_type: Type of prior ('laplace', 'sparse_gaussian', 'kurtosis')
            entropy_weight: Weight for entropy regularization
            device: Device for computation
        """
        super().__init__()

        self.model = model
        self.layer_name = layer_name
        self.channel_idx = channel_idx
        self.prior_type = prior_type
        self.entropy_weight = entropy_weight
        self.device = device

        # Variational parameters
        self.image_param = VariationalImageParameter(
            image_size=image_size,
            channels=3,
            device=device
        )

        # Prior
        self.prior = ActivationDistributionPrior()

        # Hook for extracting activations
        self.activations = None
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        """Hook to capture activations."""
        self.activations = output

    def register_hook(self):
        """Register forward hook on target layer."""
        # Find the layer
        for name, module in self.model.named_modules():
            # Convert PyTorch name to Lucent name format
            lucent_name = name.replace('.', '_')
            if lucent_name == self.layer_name:
                self._hook_handle = module.register_forward_hook(self._hook_fn)
                return

        raise ValueError(f"Layer {self.layer_name} not found in model")

    def remove_hook(self):
        """Remove forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def forward(
        self,
        n_samples: int = 16,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: sample inputs and get activation distribution.

        Args:
            n_samples: Number of samples for Monte Carlo
            temperature: Sampling temperature

        Returns:
            Tuple of (sampled_images, activations)
        """
        # Sample inputs
        images = self.image_param.sample(n_samples, temperature)

        # Forward through model
        _ = self.model(images)

        # Extract target activations
        if self.activations is None:
            raise RuntimeError("No activations captured. Did you register the hook?")

        # Get target channel activations
        activations = self.activations[:, self.channel_idx]

        return images, activations

    def compute_loss(
        self,
        activations: torch.Tensor,
        compute_entropy: bool = True,
        prior_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss: KL divergence + entropy regularization.

        Args:
            activations: Activation values from forward pass
            compute_entropy: Whether to compute entropy term
            prior_kwargs: Additional kwargs for prior

        Returns:
            Dictionary with loss components
        """
        if prior_kwargs is None:
            prior_kwargs = {}

        # KL divergence to target prior
        if self.prior_type == 'laplace':
            kl_loss = self.prior.laplace_prior(
                activations,
                target_mean=prior_kwargs.get('target_mean', 0.0),
                target_scale=prior_kwargs.get('target_scale', 1.0)
            )
        elif self.prior_type == 'sparse_gaussian':
            kl_loss = self.prior.sparse_gaussian_prior(
                activations,
                target_mean=prior_kwargs.get('target_mean', 5.0),
                target_std=prior_kwargs.get('target_std', 1.0),
                sparsity_weight=prior_kwargs.get('sparsity_weight', 0.1)
            )
        elif self.prior_type == 'kurtosis':
            kl_loss = self.prior.kurtosis_matching_loss(
                activations,
                target_kurtosis=prior_kwargs.get('target_kurtosis', 3.0)
            )
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

        losses = {
            'kl_loss': kl_loss,
            'total_loss': kl_loss
        }

        # Entropy regularization (optional, expensive)
        if compute_entropy and self.entropy_weight > 0:
            entropy = HessianEntropyEstimator.diagonal_hessian_entropy(
                kl_loss,
                self.image_param.mu
            )

            # Maximize entropy = minimize negative entropy
            entropy_loss = -self.entropy_weight * entropy

            losses['entropy'] = entropy
            losses['entropy_loss'] = entropy_loss
            losses['total_loss'] = kl_loss + entropy_loss

        return losses

    def get_visualization(self) -> torch.Tensor:
        """Get the mean image (MAP estimate)."""
        return self.image_param.get_mean()


if __name__ == "__main__":
    print("🎨 Variational Bayesian Feature Visualization")
    print("=" * 60)
    print("\nModules:")
    print("  - VariationalImageParameter: Reparameterization trick")
    print("  - ActivationDistributionPrior: Sparsity/kurtosis priors")
    print("  - HessianEntropyEstimator: Entropy via Hessian")
    print("  - VariationalFeatureVisualizer: Main class")
