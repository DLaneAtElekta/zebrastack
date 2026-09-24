import torch

from gtv.generative import AmortizedEncoder, ConvFactorAnalysis, TICAPrior, fit_free_energy, residual_log_precision
from gtv.stages import V1Stage, V2Stage, energy


def test_v1_without_normalization_is_log_energy():
    x = torch.randn(2, 1, 32, 32)
    v1 = V1Stage(n_orientations=4, n_scales=2, normalize=False)
    parts = v1.parts(x)
    e = energy(v1.bank(x))
    assert torch.allclose(parts["normalized"], e)
    assert torch.allclose(parts["log"], torch.log(e + v1.log.eps))


def test_learned_precision_is_per_channel_bounded_and_waits():
    torch.manual_seed(0)
    v1 = V1Stage(n_orientations=4, n_scales=2, decimate=2)
    with torch.no_grad():
        maps = v1(torch.randn(24, 1, 32, 32))
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), sheet=4, dim=16, whiten_groups=[8, 8],
                 second_order={"freqs": [0.0625]})
    v2.fit(maps, n_iter=30, polish_iter=5)
    enc = AmortizedEncoder(v2)
    dec = ConvFactorAnalysis(16, maps.shape[1], kernel=3)
    dec.fit_standardizer(maps)
    hist = fit_free_energy(maps, enc, dec, TICAPrior(v2.tica.h), torch.tensor(1.0), n_iter=80, batch=8,
                           learn_precision=True, free_precision_after=40, log_precision_bounds=(0.0, 2.0),
                           lr_precision=0.1, generator=torch.Generator().manual_seed(0), log_every=10)
    lp = hist["log_precision"]
    assert lp.shape == (maps.shape[1],)
    assert (lp >= 0.0).all() and (lp <= 2.0).all()
    before = [v for i, v in zip(hist["iter"], hist["log_precision_mean"]) if i < 40]
    assert all(abs(v - 1.0) < 1e-6 for v in before)  # held until freed
    assert abs(hist["log_precision_mean"][-1] - 1.0) > 1e-3  # then it moves
    assert residual_log_precision(enc, dec, maps).shape == (maps.shape[1],)
