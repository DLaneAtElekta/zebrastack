import math

import pytest
import torch

from gtv.generative import (
    AmortizedEncoder,
    ConvFactorAnalysis,
    TICAPrior,
    context_precision_fe,
    expected_signal_precision,
    residual_variance,
    settle,
)
from gtv.generative.context import accuracy_curvature
from gtv.probes.sets import SCENE_KINDS, second_order_scene
from gtv.stages import V1Stage, V2Stage


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    v1 = V1Stage(n_orientations=4, n_scales=2, decimate=2)
    with torch.no_grad():
        maps = v1(torch.randn(16, 1, 32, 32))
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), sheet=4, dim=16, whiten_groups=[8, 8],
                 second_order={"freqs": [0.0625]})
    v2.fit(maps, n_iter=30, polish_iter=5)
    enc = AmortizedEncoder(v2)
    dec = ConvFactorAnalysis(16, maps.shape[1], kernel=3)
    dec.fit_standardizer(maps)
    with torch.no_grad():
        dec.weight.normal_(std=0.3)
    return maps, enc, dec, TICAPrior(v2.tica.h)


def _f(x, mu, dec, prior, lp):
    err = (x - dec.mean(mu))[..., 2:-2, 2:-2]
    return float((0.5 * lp.exp().view(1, -1, 1, 1) * err.pow(2)).sum() + prior.neg_log(mu).sum())


def test_curvature_matches_quadratic_form(model):
    maps, enc, dec, prior = model
    lp = torch.zeros(maps.shape[1])
    lam = accuracy_curvature(dec, lp, (1, 16, 8, 8), n_iter=50)
    # no direction can have a larger Rayleigh quotient than the top eigenvalue
    zero = dec.mean(torch.zeros(1, 16, 8, 8)).detach()
    for s in range(5):
        v = torch.randn(1, 16, 8, 8, generator=torch.Generator().manual_seed(s))
        q = float((0.5 * (dec.mean(v) - zero)[..., 2:-2, 2:-2].pow(2)).sum() * 2 / v.pow(2).sum())
        assert q <= lam * 1.01


@pytest.mark.parametrize("log_pi", [0.0, 4.0])
def test_settling_lowers_free_energy_at_any_precision(model, log_pi):
    maps, enc, dec, prior = model
    x = dec.from_v1(maps)
    with torch.no_grad():
        mu0, _ = enc(maps)
    lp = torch.full((maps.shape[1],), log_pi)
    mu = settle(x, mu0, dec, prior, lp, n_steps=15, step=1.0)
    assert torch.isfinite(mu).all()
    assert _f(x, mu, dec, prior, lp) < _f(x, mu0, dec, prior, lp)


def test_context_precision_is_ml_after_settling(model):
    maps, enc, dec, prior = model
    lp = context_precision_fe(maps, enc, dec, prior, torch.zeros(maps.shape[1]), n_rounds=2, n_steps=5)
    assert lp.shape == (maps.shape[1],) and torch.isfinite(lp).all()
    # fixed point of the closed-form update: one more round moves it little
    x = dec.from_v1(maps)
    with torch.no_grad():
        mu0, _ = enc(maps)
    mu = settle(x, mu0, dec, prior, lp, n_steps=5)
    assert (-residual_variance(x, mu, dec).log() - lp).abs().max() < 0.5


def test_expected_signal_precision_only_raises_where_signal_is_added(model):
    maps, _, dec, _ = model
    base = torch.zeros(maps.shape[1])
    louder = maps.clone()
    louder[:, 0] = louder[:, 0] + 2.0 * torch.randn_like(louder[:, 0]) * maps[:, 0].std()
    lp = expected_signal_precision(louder, maps, dec, base, slice(4, 12))
    assert lp[0] > 0.5
    assert torch.allclose(lp[1:], base[1:], atol=1e-6)


def test_second_order_scene_kinds_have_matched_power():
    g = torch.Generator().manual_seed(0)
    bg = torch.randn(64, 64, generator=g)
    power = {k: torch.stack([second_order_scene(64, bg, k, g).pow(2).mean() for _ in range(40)]).mean()
             for k in SCENE_KINDS}
    assert power["target"] == pytest.approx(power["absent"], rel=0.05)
    assert power["distractor"] == pytest.approx(power["absent"], rel=0.05)
    with pytest.raises(ValueError):
        second_order_scene(64, bg, "cat", g)
    assert math.isfinite(float(power["target"]))


def test_lbfgs_settling_goes_further_than_gradient_steps(model):
    from gtv.generative import settle_lbfgs

    maps, enc, dec, prior = model
    x = dec.from_v1(maps)
    with torch.no_grad():
        mu0, _ = enc(maps)
    lp = torch.full((maps.shape[1],), 2.0)
    f_gd = _f(x, settle(x, mu0, dec, prior, lp, n_steps=10), dec, prior, lp)
    mu_lb = settle_lbfgs(x, mu0[:, :, :, :].transpose(2, 3).transpose(2, 3), dec, prior, lp, n_steps=30)
    assert torch.isfinite(mu_lb).all()
    assert _f(x, mu_lb, dec, prior, lp) < f_gd
