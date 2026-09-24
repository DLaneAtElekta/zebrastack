"""Wake-sleep for the V1 <- V2 generative path (plan Phase 3b).

Recognition q(z | x) = N(R(features(x)), diag(s^2)): the fixed Design B front
end followed by a learnable affine map, warm-started from the Phase 2
whitening + TICA. The decoder is a ``ConvFactorAnalysis``.

  wake:  x real      -> z ~ q(z|x)    -> update decoder (and refit the prior)
  sleep: z ~ p(z)    -> x ~ p(x|z)    -> update recognition to recover z from x
"""

import math

import torch
from torch import nn

from gtv.stages.v2 import V2Stage

from .decoders import ConvFactorAnalysis


class Recognition(nn.Module):
    """Learnable affine readout on the fixed Design B features."""

    def __init__(self, v2: V2Stage, init_std: float = 0.1):
        super().__init__()
        if v2.front is None:
            raise ValueError("recognition expects a Design B V2Stage")
        self.front = v2.front
        m, b = v2.tica.affine
        self.weight = nn.Parameter(m.clone())
        self.bias = nn.Parameter(b.clone())
        self.log_s = nn.Parameter(torch.full((m.shape[0],), 2 * math.log(init_std)))

    def features(self, v1: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.front(v1)

    def mean_from_features(self, f: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nd,bdhw->bnhw", self.weight, f) + self.bias.view(1, -1, 1, 1)

    def forward(self, v1: torch.Tensor) -> torch.Tensor:
        return self.mean_from_features(self.features(v1))

    def sample(self, v1: torch.Tensor, generator=None) -> torch.Tensor:
        return self.sample_from_features(self.features(v1), generator)

    def sample_from_features(self, f: torch.Tensor, generator=None) -> torch.Tensor:
        m = self.mean_from_features(f)
        return m + torch.randn(m.shape, generator=generator) * (0.5 * self.log_s).exp().view(1, -1, 1, 1)

    def features_batched(self, v1: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        return torch.cat([self.features(x) for x in v1.split(chunk)])

    def nll(self, f: torch.Tensor, z: torch.Tensor, border: int = 2) -> torch.Tensor:
        """Sleep loss: -log q(z | x) per latent element, from features f of x."""
        m = self.mean_from_features(f)
        if border:
            z, m = z[..., border:-border, border:-border], m[..., border:-border, border:-border]
        ls = self.log_s.view(1, -1, 1, 1)
        return 0.5 * ((z - m).pow(2) * torch.exp(-ls) + ls + math.log(2 * math.pi)).mean()


def fit_decoder(decoder: ConvFactorAnalysis, x: torch.Tensor, z: torch.Tensor, n_iter: int = 300,
                lr: float = 0.05, border: int = 2, batch: int = 64, generator=None) -> list[float]:
    """Phase 3a: fit the decoder to (standardized x, recognized z) pairs."""
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    hist = []
    for _ in range(n_iter):
        idx = torch.randint(len(x), (batch,), generator=generator)
        loss = decoder.nll(x[idx], z[idx], border)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return hist


def recon_r2(decoder: ConvFactorAnalysis, rec: Recognition, v1: torch.Tensor, border: int = 4) -> float:
    """Held-out R^2 of the V1 map reconstructed through z = E[q(z|x)]."""
    with torch.no_grad():
        x = decoder.from_v1(v1)
        xh = decoder.mean(rec(v1))
        x, xh = x[..., border:-border, border:-border], xh[..., border:-border, border:-border]
        return float(1 - (x - xh).pow(2).mean() / x.var())


def wake_sleep(
    decoder: ConvFactorAnalysis,
    rec: Recognition,
    prior: nn.Module,
    v1_train: torch.Tensor,
    n_iter: int = 400,
    batch: int = 32,
    lr_wake: float = 0.01,
    lr_sleep: float = 0.003,
    refit_prior_every: int | None = 50,
    tether: float = 0.0,
    init_noise_from_sleep: bool = True,
    border: int = 2,
    generator=None,
    log_every: int = 25,
) -> dict:
    """Alternate wake and sleep steps; returns a loss/diagnostic history.

    Stabilizers (plan section 9, "wake-sleep instability"):
      * ``init_noise_from_sleep`` sets q's variance to the recognition's actual
        error on fantasies before training, so the first sleep gradients are
        not scaled up by a too-small variance
      * ``refit_prior_every=None`` keeps the prior fixed (refitting it in jumps
        shifts the fantasy distribution abruptly)
      * ``tether`` pulls the recognition weights toward their TICA start,
        relative to the weights' own scale (recognition "on rails")
    """
    opt_w = torch.optim.Adam(decoder.parameters(), lr=lr_wake)
    opt_s = torch.optim.Adam(rec.parameters(), lr=lr_sleep)
    hist = {"iter": [], "wake_nll": [], "sleep_nll": [], "latent_std": []}
    # the front end is fixed, so real-data features are computed once
    f_train = rec.features_batched(v1_train)
    x_train = decoder.from_v1(v1_train)
    w0, b0 = rec.weight.detach().clone(), rec.bias.detach().clone()
    w_scale = w0.pow(2).mean()
    if refit_prior_every is None or init_noise_from_sleep:
        with torch.no_grad():
            if refit_prior_every is None:
                prior.fit(rec.mean_from_features(f_train[:256]))
            if init_noise_from_sleep:
                zf = prior.sample(64, generator)
                f = rec.features_batched(decoder.to_v1(decoder.sample(zf, generator)))
                err = (rec.mean_from_features(f) - zf)[..., border:-border, border:-border]
                rec.log_s.copy_(err.pow(2).mean((0, 2, 3)).log())
    for it in range(n_iter):
        if refit_prior_every is not None and it % refit_prior_every == 0:
            with torch.no_grad():
                sub = torch.randperm(len(v1_train), generator=generator)[:128]
                prior.fit(rec.mean_from_features(f_train[sub]))
        # wake
        idx = torch.randint(len(v1_train), (batch,), generator=generator)
        with torch.no_grad():
            z = rec.sample_from_features(f_train[idx], generator)
        wake = decoder.nll(x_train[idx], z, border)
        opt_w.zero_grad()
        wake.backward()
        opt_w.step()
        # sleep
        with torch.no_grad():
            zf = prior.sample(batch, generator)
            f = rec.features(decoder.to_v1(decoder.sample(zf, generator)))
        sleep = rec.nll(f, zf, border)
        if tether:
            sleep = sleep + tether * ((rec.weight - w0).pow(2).mean() + (rec.bias - b0).pow(2).mean()) / w_scale
        opt_s.zero_grad()
        sleep.backward()
        opt_s.step()
        if it % log_every == 0 or it == n_iter - 1:
            with torch.no_grad():
                sd = rec.mean_from_features(f_train[idx]).std((0, 2, 3))
            hist["iter"].append(it)
            hist["wake_nll"].append(wake.item())
            hist["sleep_nll"].append(sleep.item())
            hist["latent_std"].append(float(sd.median()))
    return hist
