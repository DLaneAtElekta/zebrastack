import torch

from gtv.stages import HigherStage
from gtv.thalamus import expectation


def _templates():
    torch.manual_seed(0)
    t = torch.randn(5, 8) * 0.1
    t[2, :3] += 1.0  # category 2 predicts high values on channels 0-2
    return t


def test_uniform_expectation_is_a_constant_offset():
    t = _templates()
    L = torch.randn(2, 8, 6, 6)
    out = expectation(t, 2, alpha=0.5, spatial=False)(L)
    d = t[2] - t.mean(0)
    assert torch.allclose(out, L - 0.5 * d.view(1, -1, 1, 1))


def test_spatial_expectation_explains_away_where_input_matches():
    """Ground truth: the expected pattern sits at one location, a mismatching
    pattern at another. Explaining away removes most of the match and leaves
    the mismatch (prediction error) in place."""
    t = _templates()
    mean, d = t.mean(0), t[2] - t.mean(0)
    L = mean.view(1, -1, 1, 1).repeat(1, 1, 6, 6) + 0.01 * torch.randn(1, 8, 6, 6)
    L[0, :, 1, 1] += d  # expected content here
    other = torch.zeros(8)
    other[5:] = 1.0  # a pattern the expectation does not predict
    L[0, :, 4, 4] += other
    out = expectation(t, 2, alpha=0.2)(L)
    dev_in = (L - mean.view(1, -1, 1, 1))[0]
    dev_out = (out - mean.view(1, -1, 1, 1))[0]
    assert dev_out[:, 1, 1].norm() < 0.5 * dev_in[:, 1, 1].norm()  # matched content explained away
    assert torch.allclose(dev_out[:, 4, 4], dev_in[:, 4, 4], atol=0.05)  # mismatch stays as error


def test_stage_applies_expectation_to_second_order_only():
    torch.manual_seed(0)
    st = HigherStage("V4", in_channels=4, sheet=4, radius=1, first_budget="all")
    x = torch.randn(2, 4, 8, 8)
    f0 = st.features(x)
    assert torch.equal(st.features(x, expect=lambda L: L), f0)
    f1 = st.features(x, expect=lambda L: L - 1.0)
    assert torch.equal(f1[:, :4], f0[:, :4])  # pass-through untouched
    assert torch.allclose(f1[:, 4:], f0[:, 4:] - 1.0, atol=1e-5)
