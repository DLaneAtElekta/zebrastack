import pytest
import torch

from gtv.generative import decode_down, fit_topdown
from gtv.probes.sets import clutter_scene
from gtv.stages import HigherStage
from gtv.thalamus import feature_gain


def _fitted_stage():
    torch.manual_seed(0)
    st = HigherStage("V4", in_channels=6, sheet=4, radius=1)
    x = torch.randn(40, 6, 8, 8)
    st.fit(x, n_iter=30, polish_iter=5)
    return st, x


def test_unit_gain_is_identity_and_gain_changes_features():
    st, x = _fitted_stage()
    base = st(x)
    assert torch.allclose(st(x, gain=torch.ones(st.n_second)), base, atol=1e-5)
    g = torch.ones(st.n_second)
    g[:4] = 5.0
    assert not torch.allclose(st(x, gain=g), base, atol=1e-3)


def test_features_from_outputs_is_consistent():
    st, x = _fitted_stage()
    s = st(x)
    f_hat = st.features_from_outputs(s)
    m, b = st.tica.affine
    s_again = torch.einsum("nd,bdhw->bnhw", m, f_hat) + b.view(1, -1, 1, 1)
    assert torch.allclose(s_again, s, atol=1e-3)  # the implied features reproduce the outputs


def test_feature_gain():
    templates = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(feature_gain(templates, 0, 0.0), torch.ones(3))
    g = feature_gain(templates, 0, 1.0)
    assert g[0] > 1 > g[1] and torch.isclose(g[1], g[2])


def test_topdown_decoders_shapes():
    g = torch.Generator().manual_seed(0)
    maps = {"V4": torch.randn(30, 5, 8, 8, generator=g), "PIT": torch.randn(30, 4, 4, 4, generator=g),
            "AIT": torch.randn(30, 3, 2, 2, generator=g)}
    dec = fit_topdown(maps, ["V4", "PIT", "AIT"], n_iter=20)
    out = decode_down(maps["AIT"][:2], dec, ["V4", "PIT", "AIT"])
    assert out["PIT"].shape == (2, 4, 4, 4) and out["V4"].shape == (2, 5, 8, 8)


def test_clutter_scene():
    g = torch.Generator().manual_seed(0)
    items = [torch.rand(28, 28, generator=g) for _ in range(3)]
    s = clutter_scene(items, 64, g)
    assert s.shape == (64, 64) and abs(float(s.mean())) < 1e-5


def test_feature_similarity_field_boosts_template_like_locations():
    from gtv.thalamus import feature_similarity_field

    st, x = _fitted_stage()
    # a template that favors the first half of the second-order channels
    templates = torch.zeros(2, st.n_second)
    templates[0, : st.n_second // 2] = 1.0
    field = feature_similarity_field(st, x, templates, target=0, beta=1.0)
    assert field.shape == (x.shape[0], 1, x.shape[2], x.shape[3]) and (field > 0).all()
    # where the favored channels dominate locally, the gain is higher
    from gtv.stages.gabor import energy

    f = st.log(st.norm(energy(st.bank(x))))
    f = f - f.mean(1, keepdim=True)
    favored = f[:, : st.n_second // 2].mean(1)
    favored = (favored - favored.mean((1, 2), keepdim=True)) / favored.std((1, 2), keepdim=True)  # per image, like the field
    corr = torch.corrcoef(torch.stack([favored.flatten(), field[:, 0].log().flatten()]))[0, 1]
    assert corr > 0.9
    assert torch.allclose(feature_similarity_field(st, x, templates, 0, 0.0), torch.ones_like(field))


def test_response_noise_bottleneck():
    st, x = _fitted_stage()
    base = st.features(x)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    noisy = st.features(x, noise_T=1.0, generator=g1)
    assert torch.allclose(st.features(x, noise_T=1.0, generator=g2), noisy)  # same seed, same draws
    quiet = st.features(x, noise_T=100.0, generator=torch.Generator().manual_seed(1))
    sl = st.second_order_slice
    err_noisy = (noisy[:, sl] - base[:, sl]).pow(2).mean()
    err_quiet = (quiet[:, sl] - base[:, sl]).pow(2).mean()
    assert err_quiet < err_noisy  # smaller T is noisier
    # the pass-through channels are noisy too, so information cannot bypass the bottleneck
    assert not torch.allclose(noisy[:, : st.n_first], base[:, : st.n_first])
    first_err = (noisy[:, : st.n_first] - base[:, : st.n_first]).std(0).mean()
    assert first_err == pytest.approx(float(st.tica.whitener.parts[0].scale.mean()), rel=0.3)
