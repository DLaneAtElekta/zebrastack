"""Quadrature Gabor bank and energy (plan Phase 1, block steps [1]-[2]).

Filters are built in the frequency domain, where a Gabor is a Gaussian bump
centered on its preferred wave vector. The complex (analytic) filter has only
the positive-frequency bump, so its real and imaginary outputs are the even and
odd members of a quadrature pair, and ``|response|**2`` is the phase-invariant
energy of a complex cell.

Building in frequency avoids truncating the large low-frequency kernels, and
the parameters (freq, theta, bandwidth, aspect) stay explicit so Phase 3 can
turn them into tethered parameters "on rails".
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def sigma_from_bandwidth(freq: float, bandwidth_octaves: float) -> float:
    """Spatial envelope std (pixels, along the wave vector) for a Gabor whose
    half-amplitude frequency bandwidth is ``bandwidth_octaves``."""
    b = 2.0**bandwidth_octaves
    return math.sqrt(math.log(2) / 2) * (b + 1) / (b - 1) / (math.pi * freq)


def gabor_frequency_response(
    size: tuple[int, int],
    freq: torch.Tensor,
    theta: torch.Tensor,
    sigma_u: torch.Tensor,
    aspect: float,
) -> torch.Tensor:
    """Complex-Gabor frequency responses, shape (N, H, W), real-valued.

    ``sigma_u`` is the spatial std along the wave vector; the std along the
    stripes is ``sigma_u / aspect`` (aspect < 1 elongates the filter along its
    preferred orientation). Scaled so a unit-amplitude grating at the preferred
    frequency and orientation gives energy 1, and corrected to zero DC
    (Morlet-style) so uniform luminance produces no response.
    """
    h, w = size
    fy = torch.fft.fftfreq(h).view(1, h, 1)
    fx = torch.fft.fftfreq(w).view(1, 1, w)
    freq, theta, sigma_u = (t.view(-1, 1, 1) for t in (freq, theta, sigma_u))
    ku = fx * torch.cos(theta) + fy * torch.sin(theta)
    kv = -fx * torch.sin(theta) + fy * torch.cos(theta)
    s_u = 2 * math.pi * sigma_u
    s_v = s_u / aspect

    def envelope(du):
        return torch.exp(-0.5 * ((du * s_u) ** 2 + (kv * s_v) ** 2))

    bump = envelope(ku - freq)
    dc_leak = torch.exp(-0.5 * (freq * s_u) ** 2)
    # factor 2: a cosine grating puts half its amplitude on the positive bump
    return 2 * (bump - dc_leak * envelope(ku))


class GaborBank(nn.Module):
    """L orientations x J scales of quadrature Gabor filters, applied per input
    channel. Output channels are ordered (in_channel, scale, orientation).

    Scales are octave-spaced downward from ``max_freq`` (cycles/pixel).
    """

    def __init__(
        self,
        n_orientations: int = 8,
        n_scales: int = 3,
        max_freq: float = 0.25,
        bandwidth_octaves: float = 1.2,
        aspect: float = 0.7,
        scale_ratio: float = 2.0,
    ):
        super().__init__()
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.aspect = aspect
        freqs = torch.tensor([max_freq / scale_ratio**j for j in range(n_scales)])
        thetas = torch.arange(n_orientations) * math.pi / n_orientations
        f, t = torch.meshgrid(freqs, thetas, indexing="ij")
        sig = torch.tensor([sigma_from_bandwidth(float(x), bandwidth_octaves) for x in f.flatten()])
        # fixed developmental prior for now; Phase 3 may make these tethered Parameters
        self.register_buffer("freq", f.flatten())
        self.register_buffer("theta", t.flatten())
        self.register_buffer("sigma_u", sig)
        self._cache: dict[tuple[int, int], torch.Tensor] = {}

    @property
    def n_filters(self) -> int:
        return self.freq.numel()

    def support_radius(self) -> int:
        """Pixels needed around a point to cover ~3 std of the widest envelope."""
        return math.ceil(3 * float(self.sigma_u.max()) / min(self.aspect, 1.0))

    def _response(self, size: tuple[int, int]) -> torch.Tensor:
        key = size
        g = self._cache.get(key)
        if g is None or g.device != self.freq.device:
            g = gabor_frequency_response(size, self.freq, self.theta, self.sigma_u, self.aspect)
            self._cache[key] = g
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) real -> (B, C * n_filters, H, W) complex responses."""
        b, c, h, w = x.shape
        pad_h = min(self.support_radius(), h - 1)
        pad_w = min(self.support_radius(), w - 1)
        xp = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        g = self._response(xp.shape[-2:])
        spec = torch.fft.fft2(xp).unsqueeze(2) * g  # (B, C, N, H', W')
        out = torch.fft.ifft2(spec)[..., pad_h : pad_h + h, pad_w : pad_w + w]
        return out.reshape(b, c * self.n_filters, h, w)

    def kernels(self, size: int = 33) -> torch.Tensor:
        """Spatial complex kernels (N, size, size), centered, for the filter atlas."""
        g = gabor_frequency_response((size, size), self.freq, self.theta, self.sigma_u, self.aspect)
        return torch.fft.fftshift(torch.fft.ifft2(g), dim=(-2, -1))


def energy(responses: torch.Tensor) -> torch.Tensor:
    """Complex-cell energy: squared magnitude of the quadrature pair."""
    return responses.real**2 + responses.imag**2
