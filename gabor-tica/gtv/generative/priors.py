"""Priors over V2 latent maps (B, n, h, w) for fantasies (plan Phase 3).

``IIDGaussianPrior``      standard factor-analysis prior, independent per location
``SpatialGaussianPrior``  Gaussian random field matching the latents' mean
                          spatial autocorrelation (recognized V2 latents are
                          smooth because V2 receptive fields overlap on the grid)
``StationaryGaussianPrior`` stationary Gaussian matching all lagged cross-unit
                          covariances (the full cross-spectral matrix)
``VarianceFieldPrior``    Phase 3c: dependent variances (Karklin & Lewicki
                          2009 flavor): the stationary prior's local field is
                          modulated by a per-map gain and smooth hyper-latent
                          variance fields
"""

import math

import torch
from torch import nn

from gtv.stages.tica import FittedModule

# E[log u^2] and Var[log u^2] for u ~ N(0, 1)
LOG_CHI2_MEAN = -(0.5772156649 + math.log(2))
LOG_CHI2_VAR = math.pi**2 / 2


def autocorrelation(z: torch.Tensor, max_lag: int, border: int = 2) -> torch.Tensor:
    """Mean (over units) spatial autocorrelation of latent maps, as a
    (2L+1, 2L+1) array for lags in [-L, L], from the map interior."""
    z = z[..., border : z.shape[-2] - border, border : z.shape[-1] - border]
    z = (z - z.mean((0, 2, 3), keepdim=True)) / z.std((0, 2, 3), keepdim=True)
    h, w = z.shape[-2:]
    out = torch.zeros(2 * max_lag + 1, 2 * max_lag + 1)
    for dy in range(-max_lag, max_lag + 1):
        for dx in range(-max_lag, max_lag + 1):
            a = z[..., max(0, dy) : h + min(0, dy), max(0, dx) : w + min(0, dx)]
            b = z[..., max(0, -dy) : h + min(0, -dy), max(0, -dx) : w + min(0, -dx)]
            out[dy + max_lag, dx + max_lag] = (a * b).mean()
    return out


def field_filter(acf: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Frequency-domain amplitude filter whose white-noise output has the
    given autocorrelation (embedded circularly on a ``size`` grid)."""
    h, w = size
    L = acf.shape[0] // 2
    c = torch.zeros(h, w)
    for dy in range(-L, L + 1):
        for dx in range(-L, L + 1):
            c[dy % h, dx % w] = acf[dy + L, dx + L]
    power = torch.fft.fft2(c).real.clamp(min=0)
    return power.sqrt()


def gaussian_acf(std: float, max_lag: int) -> torch.Tensor:
    t = torch.arange(-max_lag, max_lag + 1, dtype=torch.float32)
    g = torch.exp(-0.25 * (t / max(std, 1e-3)) ** 2)  # acf of a Gaussian-smoothed field
    return g[:, None] * g[None, :]


def sample_field(n: int, c: int, amp: torch.Tensor, generator=None) -> torch.Tensor:
    """(n, c, h, w) unit-variance Gaussian fields with amplitude filter ``amp``."""
    h, w = amp.shape
    eps = torch.randn(n, c, h, w, generator=generator)
    f = torch.fft.ifft2(torch.fft.fft2(eps) * amp).real
    return f / f.std((0, 2, 3), keepdim=True).clamp(min=1e-6)


class IIDGaussianPrior(nn.Module):
    def __init__(self, n: int, size: tuple[int, int]):
        super().__init__()
        self.n, self.size = n, size

    def fit(self, z: torch.Tensor) -> "IIDGaussianPrior":
        return self

    def sample(self, batch: int, generator=None) -> torch.Tensor:
        return torch.randn(batch, self.n, *self.size, generator=generator)


class SpatialGaussianPrior(FittedModule):
    """Unit-variance Gaussian fields with the latents' mean autocorrelation."""

    def __init__(self, n: int, size: tuple[int, int], max_lag: int = 5):
        super().__init__()
        self.n, self.size, self.max_lag = n, size, max_lag
        self.register_buffer("amp", torch.ones(size))
        self.register_buffer("acf", torch.zeros(2 * max_lag + 1, 2 * max_lag + 1))

    def fit(self, z: torch.Tensor) -> "SpatialGaussianPrior":
        self.acf = autocorrelation(z, self.max_lag)
        self.amp = field_filter(self.acf, self.size)
        return self

    def sample(self, batch: int, generator=None) -> torch.Tensor:
        return sample_field(batch, self.n, self.amp, generator)


def lagged_cross_covariance(z: torch.Tensor, max_lag: int, border: int = 2) -> torch.Tensor:
    """C[dy, dx] = E[z(p) z(p + (dy, dx))^T] over the map interior,
    shape (2L+1, 2L+1, n, n), after centering each unit."""
    z = z[..., border : z.shape[-2] - border, border : z.shape[-1] - border]
    z = z - z.mean((0, 2, 3), keepdim=True)
    h, w = z.shape[-2:]
    L = max_lag
    out = torch.zeros(2 * L + 1, 2 * L + 1, z.shape[1], z.shape[1])
    for dy in range(-L, L + 1):
        for dx in range(-L, L + 1):
            a = z[..., max(0, -dy) : h + min(0, -dy), max(0, -dx) : w + min(0, -dx)]
            b = z[..., max(0, dy) : h + min(0, dy), max(0, dx) : w + min(0, dx)]
            out[dy + L, dx + L] = torch.einsum("bihw,bjhw->ij", a, b) / (a.shape[0] * a.shape[2] * a.shape[3])
    return out


class StationaryGaussianPrior(FittedModule):
    """Stationary Gaussian over latent maps with a full cross-spectral matrix.

    With ``global_component`` the latents are modeled as z = a + r: a per-map
    offset vector ``a`` (one small image patch shares a lot: overall contrast,
    dominant texture) plus a stationary local field ``r``. No stationary field
    with decaying correlations can produce a per-map constant.

    Lagged cross-unit covariances of r up to ``max_lag`` are estimated from the map
    interior (recognized latents are distorted near the borders), tapered to
    zero at ``max_lag`` with a flat top over the shorter lags, and embedded on a torus of ``grid`` x ``grid`` cells,
    larger than the latent map so that sampling and cropping never sees the
    wrap-around. Negative eigenvalues of each frequency's (n x n) spectral
    matrix (from finite lags) are clipped. Sampling colors white noise with the
    per-frequency matrix square root. This is the maximum-likelihood Gaussian
    for the recognized latents, i.e. the prior's own wake-phase update.
    """

    def __init__(
        self,
        n: int,
        size: tuple[int, int],
        max_lag: int = 8,
        grid: int | None = None,
        border: int = 2,
        global_component: bool = True,
    ):
        super().__init__()
        self.n, self.size, self.max_lag, self.border = n, size, max_lag, border
        self.global_component = global_component
        self.register_buffer("global_root", torch.zeros(n, n))
        self.grid = grid or 2 * max(size)
        if self.grid < max(size) + max_lag:
            raise ValueError("grid must exceed map size + max_lag to avoid wrap-around")
        self.register_buffer("root", torch.empty(0))  # (g, g, n, n) complex
        self.register_buffer("mean", torch.zeros(n))

    def fit(self, z: torch.Tensor) -> "StationaryGaussianPrior":
        g, L, bd = self.grid, self.max_lag, self.border
        zi = z[..., bd:-bd, bd:-bd]
        self.mean = zi.mean((0, 2, 3))
        if self.global_component:
            # per-map offsets a ~ N(0, cov of per-map means); the local field is
            # fitted to what remains after removing each map's own mean
            a = zi.mean((2, 3)) - self.mean
            evals, evecs = torch.linalg.eigh(torch.cov(a.T))
            self.global_root = evecs * evals.clamp(min=0).sqrt()
            z = z - zi.mean((2, 3), keepdim=True)
        c = lagged_cross_covariance(z, L, bd)
        # Tukey taper: 1 up to max_lag / 2 (short lags unbiased), cosine to 0 past max_lag
        t = torch.arange(-L, L + 1, dtype=torch.float32).abs()
        flat = L / 2
        taper = torch.where(t <= flat, torch.ones_like(t), 0.5 * (1 + torch.cos(math.pi * (t - flat) / (L + 1 - flat))))
        c = c * (taper[:, None] * taper[None, :])[..., None, None]
        emb = torch.zeros(g, g, self.n, self.n)
        for dy in range(-L, L + 1):
            for dx in range(-L, L + 1):
                emb[dy % g, dx % g] = c[dy + L, dx + L]
        spec = torch.fft.fft2(emb.to(torch.complex64), dim=(0, 1))
        spec = 0.5 * (spec + spec.conj().transpose(-1, -2))  # enforce Hermitian
        evals, evecs = torch.linalg.eigh(spec)
        self.root = evecs * evals.clamp(min=0).sqrt().to(evecs.dtype).unsqueeze(-2)
        return self

    def sample_parts(self, batch: int, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
        """(global offsets a (B, n, 1, 1), local field r (B, n, h, w)), mean excluded."""
        g = self.grid
        h, w = self.size
        eps = torch.randn(batch, g, g, self.n, generator=generator)
        e = torch.fft.fft2(eps.to(torch.complex64), dim=(1, 2))
        colored = torch.einsum("hwij,bhwj->bhwi", self.root, e)
        r = torch.fft.ifft2(colored, dim=(1, 2)).real[:, :h, :w].permute(0, 3, 1, 2).contiguous()
        a = torch.zeros(batch, self.n, 1, 1)
        if self.global_component:
            a = (torch.randn(batch, self.n, generator=generator) @ self.global_root.T).view(batch, self.n, 1, 1)
        return a, r

    def sample(self, batch: int, generator=None) -> torch.Tensor:
        a, r = self.sample_parts(batch, generator)
        return a + r + self.mean.view(1, -1, 1, 1)


class VarianceFieldPrior(FittedModule):
    """Phase 3c, dependent variances (Karklin & Lewicki 2009 flavor):

        z = mean + a + exp(s / 2) * r,    s = gamma + B v

    on top of the global + local ``StationaryGaussianPrior`` (a, r). The local
    field's variance is modulated by a per-map gain ``gamma`` ~ N(0, tau^2)
    (some patches are busier than others) and by a few smooth hyper-latent
    fields v through loadings B (co-varying groups of units).

    Fitted by moments on the local part of the recognized latents
    (r = z - per-map mean): tau^2 is the excess variance of log per-map rms
    over what the unmodulated prior produces; for y = log r^2,
    cov(y) = B B^T + (pi^2 / 2) I, so B keeps the eigen-directions above the
    pi^2/2 floor. The width of v is chosen to match the spatial correlation
    of log r^2.
    """

    def __init__(self, n: int, size: tuple[int, int], max_rank: int = 16, border: int = 2):
        super().__init__()
        self.n, self.size, self.max_rank, self.border = n, size, max_rank, border
        self.g = StationaryGaussianPrior(n, size, border=border)
        self.register_buffer("B", torch.zeros(n, 0))
        self.register_buffer("v_amp", torch.ones(size))
        self.tau = 0.0
        self.v_std = 0.0

    @staticmethod
    def _log_rms(r: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.log(r.pow(2).mean((1, 2, 3)))

    def fit(self, z: torch.Tensor, generator=None) -> "VarianceFieldPrior":
        bd = self.border
        self.g.fit(z)
        zi = z[..., bd:-bd, bd:-bd]
        r = zi - zi.mean((2, 3), keepdim=True)
        # per-map gain: excess spread of log rms beyond the unmodulated prior's
        _, r0 = self.g.sample_parts(max(len(z), 256), generator)
        r0 = r0[..., bd:-bd, bd:-bd]
        r0 = r0 - r0.mean((2, 3), keepdim=True)
        excess = self._log_rms(r).var() - self._log_rms(r0).var()
        self.tau = float((4 * excess).clamp(min=0).sqrt())  # log-variance = 2 * log-rms
        # local co-varying variance groups
        flat = (r / r.std((0, 2, 3), keepdim=True)).permute(0, 2, 3, 1).reshape(-1, self.n)
        y = torch.log(flat.pow(2) + 1e-8)
        evals, evecs = torch.linalg.eigh(torch.cov(y.T) - LOG_CHI2_VAR * torch.eye(self.n))
        keep = (evals > 0).nonzero().flatten().flip(0)[: self.max_rank]
        self.B = evecs[:, keep] * evals[keep].sqrt()
        # smoothness of v: match lag-1 correlation of log r^2
        target = float(autocorrelation(torch.log(r.pow(2) + 1e-8), 1, border=0)[1, 2])
        best = None
        for std in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            self.v_std = std
            self.v_amp = field_filter(gaussian_acf(std, 5), self.size)
            zs = self.sample(128, generator)[..., bd:-bd, bd:-bd]
            rs = zs - zs.mean((2, 3), keepdim=True)
            got = float(autocorrelation(torch.log(rs.pow(2) + 1e-8), 1, border=0)[1, 2])
            if best is None or abs(got - target) < best[0]:
                best = (abs(got - target), std)
        self.v_std = best[1]
        self.v_amp = field_filter(gaussian_acf(self.v_std, 5), self.size)
        return self

    @property
    def rank(self) -> int:
        return self.B.shape[1]

    def sample(self, batch: int, generator=None) -> torch.Tensor:
        a, r = self.g.sample_parts(batch, generator)
        s = self.tau * torch.randn(batch, 1, 1, 1, generator=generator)
        if self.rank:
            v = sample_field(batch, self.rank, self.v_amp, generator)
            s = s + torch.einsum("nk,bkhw->bnhw", self.B, v)
        # center the modulation so exp(s / 2) has unit mean power per unit
        s = s - torch.log(torch.exp(s).mean((0, 2, 3), keepdim=True))
        return a + torch.exp(0.5 * s) * r + self.g.mean.view(1, -1, 1, 1)
