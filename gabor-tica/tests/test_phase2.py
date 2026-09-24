import math

import pytest
import torch

from gtv.probes import auc, junction, linear_decode
from gtv.stages import TICA, V1Stage, V2Stage, Whitener, arrange_on_sheet, build_stage, torus_neighborhood
from gtv.stages.tica import torus_distance


def _grouped_sources(n=8000, d=16, group=4, seed=0):
    """Sparse sources whose variances are shared within groups of ``group``."""
    g = torch.Generator().manual_seed(seed)
    var = torch.exp(torch.randn(n, d // group, generator=g)).repeat_interleave(group, 1)
    lap = -torch.log(torch.rand(n, d, generator=g)) * torch.sign(torch.rand(n, d, generator=g) - 0.5)
    src = lap * var.sqrt()
    mix = torch.randn(d, d, generator=g)
    return src, src @ mix.T


def _match(s, src):
    n = s.shape[1]
    c = torch.corrcoef(torch.cat([s, src], 1).T)[:n, n:].abs()
    return c.max(1).values, c.argmax(1)


def test_junction_kinds():
    line, ell = junction(64, "line"), junction(64, "L")
    # line and L have the same ink; they differ only in configuration
    assert (line + 1).sum() == pytest.approx((ell + 1).sum(), rel=0.02)
    assert not torch.allclose(line, ell)
    assert (junction(64, "X") + 1).sum() > (junction(64, "T") + 1).sum() > (line + 1).sum()


def test_torus_neighborhood():
    h = torus_neighborhood(4, 4, 1)
    assert torch.equal(h, h.T)
    assert (h.sum(1) == 9).all()  # 3x3 including self, wrapping at edges
    assert torus_distance(4, 4)[0, 3] == 1  # opposite edges are adjacent


def test_whitener_whitens_and_affine_matches():
    x = torch.randn(4000, 10) @ torch.randn(10, 10) + 3.0
    w = Whitener().fit(x, 6)
    z = w.transform(x)
    assert torch.allclose(torch.cov(z.T), torch.eye(6), atol=0.05)
    assert torch.allclose(x @ w.matrix.T + w.offset, z, atol=1e-4)


def test_arrange_on_sheet_improves_objective():
    torch.manual_seed(0)
    s = torch.randn(2000, 16) * torch.exp(torch.randn(2000, 4)).repeat_interleave(4, 1)
    s = s[:, torch.randperm(16)]
    h = torus_neighborhood(4, 4, 1)
    hh = h - torch.eye(16)
    e = torch.corrcoef(s.pow(2).T).fill_diagonal_(0)
    perm = arrange_on_sheet(s, h)
    assert sorted(perm.tolist()) == list(range(16))
    assert (hh * e[perm][:, perm]).sum() > (hh * e).sum()


@pytest.mark.parametrize("mode,kw", [("complete", dict(n_iter=300)), ("rica", dict(n_iter=1500, lam=2.0))])
def test_tica_recovers_sources_and_groups_them(mode, kw):
    src, x = _grouped_sources()
    t = TICA(4, 4, radius=1)
    hist = t.fit(x, 16, mode=mode, **kw)
    assert hist[-1] < hist[0]
    assert torch.allclose(t.weight.norm(dim=1), torch.ones(16), atol=1e-4)  # no collapsed units
    best, which = _match(t(x), src)
    assert best.min() > 0.95  # every unit is one source
    assert len(set(which.tolist())) == 16
    grp = which // 4
    d = torus_distance(4, 4)
    same = (grp[:, None] == grp[None, :]) & ~torch.eye(16, dtype=torch.bool)
    assert d[same].float().mean() < d[~same & ~torch.eye(16, dtype=torch.bool)].float().mean()


def test_second_order_keeps_frequency_decreasing_paths():
    v2 = V2Stage(design="B", v1_freqs=[0.25, 0.125, 0.0625], second_order={"freqs": [0.0625, 0.03125]})
    assert v2.front.keep == [[0, 1], [0, 1, 2]]
    assert v2.front.n_channels == 3 + 5 * 4


@pytest.fixture(scope="module")
def fitted():
    v1 = V1Stage(n_orientations=4, n_scales=2, decimate=2)
    torch.manual_seed(0)
    maps = v1(torch.randn(24, 1, 32, 32))
    kw = dict(sheet=4, dim=16, tica_mode="complete")
    a = V2Stage(design="A", patch=4, **kw)
    b = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), second_order={"freqs": [0.0625]}, **kw)
    a.fit(maps, n_per_image=16, n_iter=50, polish_iter=10)
    b.fit(maps, n_iter=50, polish_iter=10)
    return maps, a, b


def test_design_b_forward_matches_tica_on_feature_vectors(fitted):
    maps, _, b = fitted
    s = b(maps)
    f = b.features(maps)
    i, y, x = 3, 5, 7
    assert torch.allclose(s[i, :, y, x], b.tica(f[i, :, y, x][None])[0], atol=1e-4)


def test_design_a_conv_matches_tica_on_patches(fitted):
    maps, a, _ = fitted
    s = a(maps)
    assert s.shape == (24, 16, 8, 8)
    i, y, x = 2, 4, 3
    pad = (a.patch - 2) // 2
    y0, x0 = 2 * y - pad, 2 * x - pad
    patch = maps[i, :, y0 : y0 + a.patch, x0 : x0 + a.patch].reshape(1, -1)
    assert torch.allclose(s[i, :, y, x], a.tica(patch)[0], atol=1e-4)


def test_v2_registered_and_pooled_energy(fitted):
    maps, _, b = fitted
    st = build_stage({"name": "V2", "kind": "v2_tica", "design": "A", "sheet": 4})
    assert isinstance(st, V2Stage)
    s = b(maps)
    e = b.pooled_energy(s)
    assert e.shape == s.shape and (e > 0).all()


def test_decoding_helpers():
    assert auc(torch.tensor([0.9, 0.8, 0.1, 0.2]), torch.tensor([1, 1, 0, 0])) == 1.0
    assert auc(torch.tensor([0.1, 0.2, 0.9, 0.8]), torch.tensor([1, 1, 0, 0])) == 0.0
    g = torch.Generator().manual_seed(0)
    y = torch.arange(3).repeat_interleave(50)
    x = torch.randn(150, 5, generator=g) + 3 * torch.nn.functional.one_hot(y, 5)
    assert linear_decode(x, y) > 0.9
    assert linear_decode(torch.randn(150, 5, generator=g), y) < 0.6
