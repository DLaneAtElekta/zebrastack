import torch

from gtv.data import affine, center_crop, drift_pairs, load_grayscale, REPO_PHOTOS
from gtv.stages import torus_neighborhood
from gtv.stages.tica import tica_loss
from gtv.temporal import bubbles_loss, fit_temporal

import pytest


def _impulse(y=16, x=16, size=32):
    img = torch.zeros(1, 1, size, size)
    img[0, 0, y, x] = 1.0
    return img


def _peak(img):
    i = int(img[0, 0].flatten().argmax())
    return divmod(i, img.shape[-1])


def test_affine_shift_rotation_scale():
    one, zero = torch.ones(1), torch.zeros(1)
    x = _impulse()
    assert torch.allclose(affine(x, torch.zeros(1, 2), zero, one), x, atol=1e-6)
    assert _peak(affine(x, torch.tensor([[3.0, 0.0]]), zero, one)) == (16, 19)  # content moves right
    assert _peak(affine(x, torch.tensor([[0.0, 2.0]]), zero, one)) == (18, 16)  # and down
    off = _impulse(16, 24)
    y, xcol = _peak(affine(off, torch.zeros(1, 2), zero, torch.full((1,), 1.25)))
    assert xcol > 24  # zooming in pushes off-center content outward


@pytest.mark.skipif(not REPO_PHOTOS.exists(), reason="repo photos not present")
def test_drift_pairs_are_correlated_but_different():
    p = drift_pairs(load_grayscale(limit=2, max_side=256), 8, 32, generator=torch.Generator().manual_seed(0))
    assert p.shape == (8, 2, 1, 32, 32)
    a, b = p[:, 0].flatten(), p[:, 1].flatten()
    c = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    assert 0.2 < c < 0.999
    assert center_crop(torch.zeros(1, 1, 10, 10), 4).shape[-1] == 4


def test_bubbles_loss_rewards_persistent_energy():
    h = torus_neighborhood(2, 2, 0)  # radius 0: each unit pools only itself
    s_t = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    same = torch.tensor([[1.0, 0.0, 0.0, 0.0]])    # energy stays in unit 0
    hop = torch.tensor([[0.0, 1.0, 0.0, 0.0]])     # energy hops to unit 1
    assert bubbles_loss(s_t, same, h, 1.0) < bubbles_loss(s_t, hop, h, 1.0)
    # weight 0 is still TICA on both frames
    assert torch.isclose(bubbles_loss(s_t, hop, h, 0.0), tica_loss(s_t, h) + tica_loss(hop, h))


def test_linear_slowness_is_rotation_invariant():
    g = torch.Generator().manual_seed(0)
    z_t, z_t1 = torch.randn(500, 6, generator=g), torch.randn(500, 6, generator=g)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=g))
    lin = lambda w: ((z_t1 - z_t) @ w.T).pow(2).sum(1).mean()  # noqa: E731
    assert torch.isclose(lin(torch.eye(6)), lin(q), rtol=1e-4)  # why linear slowness cannot steer W


@pytest.mark.parametrize("mode,weight", [("bubbles", 1.0), ("coherence", 1.0)])
def test_temporal_fits_recover_persistent_energies(mode, weight):
    g = torch.Generator().manual_seed(0)
    n, d = 4000, 8
    # sources whose amplitude persists across frames (to a varying degree) while the sign does not
    amp_rho = torch.linspace(0.0, 0.95, d)
    a0 = torch.randn(n, d, generator=g)
    a1 = amp_rho * a0 + (1 - amp_rho**2).sqrt() * torch.randn(n, d, generator=g)
    src_t = a0.abs() * torch.sign(torch.randn(n, d, generator=g))
    src_t1 = a1.abs() * torch.sign(torch.randn(n, d, generator=g))
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    z_t, z_t1 = src_t @ q.T, src_t1 @ q.T
    h = torus_neighborhood(2, 4, 0)

    def energy_corr(w):
        e0, e1 = (z_t @ w.T).pow(2), (z_t1 @ w.T).pow(2)
        return float(torch.stack([torch.corrcoef(torch.stack([e0[:, i], e1[:, i]]))[0, 1] for i in range(d)]).mean())

    w_still, _ = fit_temporal(z_t, z_t1, h, "still", n_iter=150, arrange=False)
    w_slow, _ = fit_temporal(z_t, z_t1, h, mode, weight, n_iter=150, arrange=False)
    assert energy_corr(w_slow) > energy_corr(w_still) + 0.1
    assert torch.allclose(w_slow @ w_slow.T, torch.eye(d), atol=1e-4)
