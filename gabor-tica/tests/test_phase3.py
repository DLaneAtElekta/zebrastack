import math

import pytest
import torch

from gtv.generative import (
    ConvFactorAnalysis,
    StationaryGaussianPrior,
    VarianceFieldPrior,
    fit_decoder,
    statistic_gap,
)
from gtv.generative.priors import autocorrelation, field_filter, gaussian_acf, lagged_cross_covariance, sample_field
from gtv.viz import orientation_energy_image


def _smooth_latents(n_maps=96, n=6, size=16, std=1.5, offset_sd=0.6, seed=0):
    """Latent maps = mixed smooth fields + a per-map offset, like recognized V2 latents."""
    g = torch.Generator().manual_seed(seed)
    f = sample_field(n_maps, n, field_filter(gaussian_acf(std, 6), (size, size)), g)
    mix = torch.linalg.qr(torch.randn(n, n, generator=g))[0]
    z = torch.einsum("ij,bjhw->bihw", mix, f)
    return z + offset_sd * torch.randn(n_maps, n, 1, 1, generator=g)


def test_decoder_upsamples_and_recovers_linear_generator():
    g = torch.Generator().manual_seed(0)
    true = ConvFactorAnalysis(4, 3, kernel=3)
    with torch.no_grad():
        true.weight.normal_(generator=g)
    z = torch.randn(64, 4, 8, 8, generator=g)
    with torch.no_grad():
        x = true.mean(z) + 0.05 * torch.randn(64, 3, 16, 16, generator=g)
    dec = ConvFactorAnalysis(4, 3, kernel=3)
    assert dec.mean(z).shape == (64, 3, 16, 16)
    hist = fit_decoder(dec, x, z, n_iter=300, lr=0.05, border=1, generator=g)
    assert hist[-1] < hist[0]
    with torch.no_grad():
        r2 = 1 - (x - dec.mean(z)).pow(2).mean() / x.var()
    assert r2 > 0.98
    assert dec.log_psi.exp().max() < 0.2  # noise variance moves from 1 toward the true 0.05**2


def test_standardizer_round_trip():
    v1 = torch.randn(8, 3, 10, 10) * torch.tensor([1.0, 5.0, 0.1]).view(1, 3, 1, 1) + 2
    dec = ConvFactorAnalysis(2, 3)
    dec.fit_standardizer(v1)
    x = dec.from_v1(v1)
    assert torch.allclose(x.std((0, 2, 3)), torch.ones(3), atol=1e-4)
    assert torch.allclose(dec.to_v1(x), v1, atol=1e-4)


def test_stationary_prior_matches_lagged_cross_covariance_and_offsets():
    z = _smooth_latents()
    prior = StationaryGaussianPrior(6, (16, 16), max_lag=6).fit(z)
    s = prior.sample(512, torch.Generator().manual_seed(1))
    zi, si = z[..., 2:-2, 2:-2], s[..., 2:-2, 2:-2]
    c_real = lagged_cross_covariance(z, 2)
    c_samp = lagged_cross_covariance(s, 2)
    # a 96-map estimate is itself ~0.15 from the truth, so this is at the noise floor
    assert (c_samp - c_real).norm() / c_real.norm() < 0.15
    # the global component reproduces per-map offset variance
    assert si.mean((2, 3)).var(0).mean() == pytest.approx(zi.mean((2, 3)).var(0).mean(), rel=0.35)
    no_global = StationaryGaussianPrior(6, (16, 16), max_lag=6, global_component=False).fit(z)
    s2 = no_global.sample(512, torch.Generator().manual_seed(1))[..., 2:-2, 2:-2]
    assert s2.mean((2, 3)).var(0).mean() < 0.75 * zi.mean((2, 3)).var(0).mean()  # underestimates offsets


def test_variance_prior_detects_per_map_gain():
    g = torch.Generator().manual_seed(0)
    z = _smooth_latents(offset_sd=0.0, n_maps=128)
    gain = torch.exp(0.6 * torch.randn(128, 1, 1, 1, generator=g))
    with_gain = VarianceFieldPrior(6, (16, 16)).fit(z * gain, generator=g)
    without = VarianceFieldPrior(6, (16, 16)).fit(z, generator=g)
    assert with_gain.tau > 0.6
    assert without.tau < 0.35
    s = with_gain.sample(64, g)
    assert s.shape == (64, 6, 16, 16) and torch.isfinite(s).all()


def test_statistic_gap_floor_and_ceiling():
    g = torch.Generator().manual_seed(0)
    real = sample_field(64, 6, field_filter(gaussian_acf(2.0, 6), (24, 24)), g)
    same = sample_field(64, 6, field_filter(gaussian_acf(2.0, 6), (24, 24)), g)
    noise = torch.randn(64, 6, 24, 24, generator=g)
    floor = statistic_gap(real, same, 3, 2)["overall"]
    ceiling = statistic_gap(real, noise, 3, 2)["overall"]
    assert floor < 0.4 and ceiling > 5 * floor


def test_prior_state_round_trip(tmp_path):
    z = _smooth_latents(n_maps=32)
    p = StationaryGaussianPrior(6, (16, 16)).fit(z)
    torch.save(p.state_dict(), tmp_path / "p.pt")
    q = StationaryGaussianPrior(6, (16, 16))
    q.load_state_dict(torch.load(tmp_path / "p.pt"))
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.allclose(p.sample(4, g1), q.sample(4, g2), atol=1e-5)


def test_orientation_energy_image():
    m = torch.zeros(6, 10, 10)
    m[0] = 3.0  # scale 0, orientation 0 dominates
    img = orientation_energy_image(m, 3, 2, scale=0)
    assert img.shape == (10, 10, 3)
    assert (img >= 0).all() and (img <= 1).all()
    assert autocorrelation(torch.randn(4, 2, 12, 12), 1).shape == (3, 3)
    assert math.isfinite(float(img.sum()))


def test_wake_sleep_runs_stably_on_a_tiny_model():
    from gtv.generative import Recognition, wake_sleep
    from gtv.stages import V1Stage, V2Stage

    torch.manual_seed(0)
    v1 = V1Stage(n_orientations=4, n_scales=2, decimate=2)
    with torch.no_grad():
        maps = v1(torch.randn(32, 1, 32, 32))
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), sheet=4, dim=16, whiten_groups=[8, 8],
                 second_order={"freqs": [0.0625]})
    v2.fit(maps, n_iter=30, polish_iter=5)
    rec = Recognition(v2)
    w0 = rec.weight.detach().clone()
    dec = ConvFactorAnalysis(16, maps.shape[1], kernel=3)
    dec.fit_standardizer(maps)
    prior = StationaryGaussianPrior(16, (8, 8), max_lag=3)
    hist = wake_sleep(dec, rec, prior, maps, n_iter=15, batch=8, refit_prior_every=None, tether=1.0,
                      generator=torch.Generator().manual_seed(0), log_every=5)
    assert all(math.isfinite(v) for v in hist["wake_nll"] + hist["sleep_nll"])
    assert rec.log_s.abs().max() > 0  # noise initialized from the measured sleep error
    assert (rec.weight - w0).norm() / w0.norm() < 0.5  # tethered near the TICA start
