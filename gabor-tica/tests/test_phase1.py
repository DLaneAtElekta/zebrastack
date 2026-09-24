import math

import pytest
import torch

from gtv.data import REPO_PHOTOS, load_grayscale, random_patches
from gtv.probes import (
    center,
    excess_kurtosis,
    grating,
    grating_responses,
    half_width_half_height,
    phase_modulation,
    skewness,
)
from gtv.stages import GaborBank, V1Stage, build_stage, energy
from gtv.stages.normalize import gaussian_blur

SIZE = 64


@pytest.fixture(scope="module")
def v1():
    return V1Stage()


def _energy(v1):
    return lambda x: v1.parts(x)["energy"]


def _normalized(v1):
    return lambda x: v1.parts(x)["normalized"]


def test_unit_grating_gives_unit_energy_and_dc_gives_none(v1):
    bank = v1.bank
    for c in (0, 11, 22):
        x = grating(SIZE, float(bank.freq[c]), float(bank.theta[c])).view(1, 1, SIZE, SIZE)
        assert center(energy(bank(x)))[0, c] == pytest.approx(1.0, abs=0.02)
    assert energy(bank(torch.ones(1, 1, SIZE, SIZE))).abs().max() < 1e-8


def test_kernels_are_quadrature_pairs():
    k = GaborBank(n_orientations=4, n_scales=2).kernels(33)
    even, odd = k.real, k.imag
    assert torch.allclose(even, even.flip(-1, -2), atol=1e-6)
    assert torch.allclose(odd, -odd.flip(-1, -2), atol=1e-6)
    assert (even * odd).sum((-1, -2)).abs().max() < 1e-5
    assert even.sum((-1, -2)).abs().max() < 1e-5  # zero DC


def test_orientation_peaks_match_nominal(v1):
    bank = v1.bank
    L = bank.n_orientations
    thetas = torch.arange(2 * L) * math.pi / (2 * L)
    r = grating_responses(_normalized(v1), SIZE, bank.freq[::L], thetas, n_phases=2)
    for c in range(bank.n_filters):
        assert torch.isclose(thetas[r[c // L, :, c].argmax()], bank.theta[c])


def test_sf_peaks_match_nominal(v1):
    bank = v1.bank
    freqs = torch.logspace(math.log10(0.02), math.log10(0.45), 60)
    for c in (0, 8, 16):
        curve = grating_responses(_energy(v1), SIZE, freqs, bank.theta[c : c + 1], n_phases=2)[:, 0, c]
        assert abs(math.log2(float(freqs[curve.argmax()] / bank.freq[c]))) < 0.1


def test_complex_cells_are_phase_invariant(v1):
    pm = phase_modulation(_normalized(v1), SIZE, 0.125, 0.0)
    assert pm[8] < 0.01


def test_contrast_invariant_shape_with_compressed_amplitude(v1):
    thetas = torch.arange(16) * math.pi / 16
    f = v1.bank.freq[8:9]
    curves = {
        k: torch.stack([grating_responses(fn, SIZE, f, thetas, contrast=c, n_phases=2)[0, :, 8] for c in (0.1, 1.0)])
        for k, fn in (("e", _energy(v1)), ("n", _normalized(v1)))
    }
    lo, hi = curves["n"]
    assert torch.corrcoef(torch.stack([lo / lo.max(), hi / hi.max()]))[0, 1] > 0.99
    raw_ratio = float(curves["e"][1].max() / curves["e"][0].max())
    norm_ratio = float(hi.max() / lo.max())
    assert raw_ratio == pytest.approx(100, rel=1e-3)
    assert norm_ratio < 0.1 * raw_ratio


def test_stage_shapes_and_registry():
    x = torch.randn(2, 1, 32, 32)
    s = build_stage({"name": "V1", "kind": "v1_gabor", "n_orientations": 4, "n_scales": 2, "decimate": 2})
    y = s(x)
    assert y.shape == (2, s.out_channels(1), 16, 16) == (2, 8, 16, 16)
    assert torch.isfinite(y).all()


def test_gaussian_blur_preserves_constants():
    x = torch.full((1, 3, 16, 16), 2.5)
    assert torch.allclose(gaussian_blur(x, 3.0), x, atol=1e-5)
    assert torch.allclose(gaussian_blur(x, 20.0), x, atol=1e-5)  # kernel wider than image


def test_hwhh_triangle():
    xs = torch.arange(11, dtype=torch.float32)
    curve = 5 - (xs - 5).abs()  # peak 5, min 0, half-height 2.5 reached 2.5 away
    assert half_width_half_height(xs, curve) == pytest.approx(2.5)


@pytest.mark.skipif(not REPO_PHOTOS.exists(), reason="repo photos not present")
def test_log_normalized_energy_is_near_gaussian(v1):
    imgs = load_grayscale(max_side=512)
    x = random_patches(imgs, 64, SIZE, torch.Generator().manual_seed(0))
    parts = v1.parts(x)
    m = slice(16, 48)
    raw = torch.tensor([excess_kurtosis(parts["energy"][:, c, m, m]) for c in range(24)]).median()
    lg = parts["log"][:, :, m, m]
    sk = torch.tensor([skewness(lg[:, c]) for c in range(24)]).median()
    ku = torch.tensor([excess_kurtosis(lg[:, c]) for c in range(24)]).median()
    assert raw > 3
    assert abs(sk) < 1 and abs(ku) < 1
