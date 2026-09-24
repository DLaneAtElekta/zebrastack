"""Top-down decoders from V2 latents to V1 log-power maps (plan Phase 3).

Generates V1 power maps only, not pixels (plan section 9: phase loss makes
pixel generation hard).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from gtv.stages.tica import FittedModule


class ConvFactorAnalysis(FittedModule):
    """Convolutional factor analysis: x = mu + G * z + noise.

    ``z`` is a latent map (B, n_latent, h, w). ``G`` upsamples it 2x
    (bilinear) and applies a ``kernel`` x ``kernel`` convolution, so each
    latent location projects onto a patch of the (2h, 2w) V1 map. (A stride-2
    transposed convolution leaves period-2 checkerboard residuals.) Noise is
    Gaussian with one variance per V1 channel. Works in per-channel standardized V1 units
    (``fit_standardizer``); ``to_v1``/``from_v1`` convert.
    """

    def __init__(self, n_latent: int, n_v1: int, kernel: int = 5):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        self.kernel = kernel
        self.weight = nn.Parameter(torch.zeros(n_v1, n_latent, kernel, kernel))
        self.mu = nn.Parameter(torch.zeros(n_v1))
        self.log_psi = nn.Parameter(torch.zeros(n_v1))
        self.register_buffer("x_mean", torch.zeros(n_v1))
        self.register_buffer("x_std", torch.ones(n_v1))
        self.register_buffer("noise_amp", torch.empty(0))  # spatial filter for sampled noise

    def fit_standardizer(self, v1: torch.Tensor) -> None:
        self.x_mean = v1.mean((0, 2, 3))
        self.x_std = v1.std((0, 2, 3))

    def from_v1(self, v1: torch.Tensor) -> torch.Tensor:
        return (v1 - self.x_mean.view(1, -1, 1, 1)) / self.x_std.view(1, -1, 1, 1)

    def to_v1(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.x_std.view(1, -1, 1, 1) + self.x_mean.view(1, -1, 1, 1)

    def mean(self, z: torch.Tensor) -> torch.Tensor:
        """Predicted standardized V1 map (B, n_v1, 2h, 2w)."""
        up = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        up = F.pad(up, (self.kernel // 2,) * 4, mode="replicate")
        return F.conv2d(up, self.weight) + self.mu.view(1, -1, 1, 1)

    def fit_noise_field(self, x: torch.Tensor, z: torch.Tensor, max_lag: int = 6) -> None:
        """Fit the spatial autocorrelation of residuals x - mean(z) so sampled
        noise is as smooth as the real residuals (V1 maps are very smooth;
        white per-cell noise would be an easy giveaway)."""
        from .priors import autocorrelation, field_filter

        with torch.no_grad():
            r = x - self.mean(z)
            self.noise_amp = field_filter(autocorrelation(r, max_lag, border=4), tuple(x.shape[-2:]))

    def sample(self, z: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        m = self.mean(z)
        if self.noise_amp.numel():
            from .priors import sample_field

            noise = sample_field(m.shape[0], m.shape[1], self.noise_amp, generator)
        else:
            noise = torch.randn(m.shape, generator=generator)
        return m + noise * (0.5 * self.log_psi).exp().view(1, -1, 1, 1)

    def nll(self, x: torch.Tensor, z: torch.Tensor, border: int = 0) -> torch.Tensor:
        """Mean Gaussian negative log-likelihood per V1 map element."""
        m = self.mean(z)
        if border:
            x, m = x[..., border:-border, border:-border], m[..., border:-border, border:-border]
        lp = self.log_psi.view(1, -1, 1, 1)
        return 0.5 * ((x - m).pow(2) * torch.exp(-lp) + lp + math.log(2 * math.pi)).mean()
