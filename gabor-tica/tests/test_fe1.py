import math

import pytest
import torch

from gtv.generative import (
    AmortizedEncoder,
    ConvFactorAnalysis,
    TICAPrior,
    active_units,
    fit_free_energy,
    free_energy,
)
from gtv.generative.free_energy import rearrange_units
from gtv.stages import V1Stage, V2Stage, torus_neighborhood


@pytest.fixture(scope="module")
def small():
    torch.manual_seed(0)
    v1 = V1Stage(n_orientations=4, n_scales=2, decimate=2)
    with torch.no_grad():
        maps = v1(torch.randn(24, 1, 32, 32))
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), sheet=4, dim=16, whiten_groups=[8, 8],
                 second_order={"freqs": [0.0625]})
    v2.fit(maps, n_iter=30, polish_iter=5)
    return maps, v2


def test_tica_prior_value():
    h = torus_neighborhood(2, 2, 1)
    p = TICAPrior(h, eps=1e-2)
    z = torch.zeros(3, 4, 5, 5)
    assert torch.allclose(p.neg_log(z), torch.full((3,), 4 * 25 * math.sqrt(1e-2)))
    z[:, 0] = 1.0  # with a 2x2 torus and radius 1, every unit pools every unit
    assert torch.allclose(p.neg_log(z), torch.full((3,), 4 * 25 * math.sqrt(1 + 1e-2)))


def test_random_encoder_init_is_whitened(small):
    maps, v2 = small
    enc = AmortizedEncoder(v2, init="random", seed=1)
    tica = AmortizedEncoder(v2, init="tica")
    f = enc.features(maps)
    with torch.no_grad():
        mu_r = enc.from_features(f)[0][..., 2:-2, 2:-2].permute(0, 2, 3, 1).reshape(-1, 16)
        mu_t = tica.from_features(f)[0][..., 2:-2, 2:-2].permute(0, 2, 3, 1).reshape(-1, 16)
    # both are rotations of the same whitened features: same total variance, different axes
    assert torch.cov(mu_r.T).trace() == pytest.approx(torch.cov(mu_t.T).trace(), rel=1e-3)
    assert not torch.allclose(mu_r, mu_t, atol=1e-2)


def test_free_energy_terms_and_precision(small):
    maps, v2 = small
    enc = AmortizedEncoder(v2)
    dec = ConvFactorAnalysis(16, maps.shape[1], kernel=3)
    dec.fit_standardizer(maps)
    x, f = dec.from_v1(maps), enc.features(maps)
    prior = TICAPrior(v2.tica.h)
    with torch.no_grad():
        t = free_energy(x, f, enc, dec, prior, torch.tensor(0.0), torch.Generator().manual_seed(0))
    assert torch.allclose(t["F"], t["accuracy"] + t["prior"] - t["entropy"])
    # entropy of N(mu, sigma^2) summed over latents: sigma is init_sigma = 0.1 everywhere
    mu, _ = enc.from_features(f)
    n_lat = mu[0].numel()  # 16 units x 8 x 8 map
    assert t["entropy"][0].item() == pytest.approx(n_lat * (math.log(0.1) + 0.5 * math.log(2 * math.pi * math.e)), rel=1e-4)
    # per-channel precision equal to a scalar gives the same accuracy
    with torch.no_grad():
        t2 = free_energy(x, f, enc, dec, prior, torch.full((maps.shape[1],), 0.0), torch.Generator().manual_seed(0))
    assert torch.allclose(t["accuracy"], t2["accuracy"])


def test_rearrange_units_preserves_reconstruction(small):
    maps, v2 = small
    enc = AmortizedEncoder(v2, init="random", seed=2)
    dec = ConvFactorAnalysis(16, maps.shape[1], kernel=3)
    with torch.no_grad():
        dec.weight.normal_()
    f = enc.features(maps)
    with torch.no_grad():
        before = dec.mean(enc.from_features(f)[0])
    perm = rearrange_units(enc, dec, TICAPrior(v2.tica.h), f)
    with torch.no_grad():
        after = dec.mean(enc.from_features(f)[0])
    assert sorted(perm.tolist()) == list(range(16))
    assert torch.allclose(before, after, atol=1e-4)


def test_fit_free_energy_decreases_and_reports_active_units(small):
    maps, v2 = small
    enc = AmortizedEncoder(v2)
    dec = ConvFactorAnalysis(16, maps.shape[1], kernel=3)
    dec.fit_standardizer(maps)
    hist = fit_free_energy(maps, enc, dec, TICAPrior(v2.tica.h), torch.tensor(3.0), n_iter=60, batch=8,
                           warmup=20, arrange_every=30, generator=torch.Generator().manual_seed(0), log_every=10)
    assert hist["F_per_element"][-1] < hist["F_per_element"][0]
    act = active_units(enc, maps)
    assert act.dtype == torch.bool and act.shape == (16,)
