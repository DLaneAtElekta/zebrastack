import math

import pytest
import torch

from gtv.probes import (
    contrast_modulated_grating,
    curvature_fragment,
    drifting_grating,
    grating,
    oriented_noise,
    texture_pair,
)
from gtv.stages import IdentityStage, build_stack
from gtv.viz import fantasy_grid, filter_atlas, sheet_map

SIZE = 64


def _peak_orientation(img: torch.Tensor) -> float:
    """Orientation (mod pi) of the strongest non-DC Fourier component."""
    spec = torch.fft.fft2(img).abs()
    spec[0, 0] = 0
    iy, ix = divmod(int(spec.argmax()), img.shape[1])
    fy = torch.fft.fftfreq(img.shape[0])[iy]
    fx = torch.fft.fftfreq(img.shape[1])[ix]
    return float(torch.remainder(torch.atan2(fy, fx), math.pi))


def _ang_dist(a: float, b: float) -> float:
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


@pytest.mark.parametrize("theta", [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
def test_grating_orientation(theta):
    # freq chosen to land on an FFT bin at the cardinal and diagonal angles
    g = grating(SIZE, 8 / SIZE, theta)
    assert g.shape == (SIZE, SIZE)
    assert g.abs().max() <= 1.0 + 1e-6
    assert _ang_dist(_peak_orientation(g), theta) < 0.1


def test_drifting_grating_moves():
    frames = drifting_grating(SIZE, 0.1, 0.0, n_frames=3, speed=2.0)
    assert frames.shape == (3, SIZE, SIZE)
    # pattern shifts 2 px along +x per frame (compare away from the wrapped edge)
    assert torch.allclose(frames[1][:, 2:], frames[0][:, :-2], atol=1e-4)


def test_oriented_noise_orientation():
    gen = torch.Generator().manual_seed(0)
    n = oriented_noise(SIZE, math.pi / 3, generator=gen, bandwidth=0.15)
    assert _ang_dist(_peak_orientation(n), math.pi / 3) < 0.35


def test_second_order_has_no_first_order_envelope():
    gen = torch.Generator().manual_seed(0)
    env_f = 2 / SIZE
    img = contrast_modulated_grating(SIZE, env_f, 0.0, carrier_freq=0.25, generator=gen)
    spec = torch.fft.fft2(img).abs()
    carrier_energy = spec.max()
    # luminance at the envelope frequency is small relative to the carrier
    assert spec[0, 2] < 0.2 * carrier_energy
    # but the rectified image carries the envelope
    rect = torch.fft.fft2(img**2).abs()
    assert rect[0, 2] > 5 * rect[0, 7]


def test_texture_pair_mask():
    gen = torch.Generator().manual_seed(0)
    img, mask = texture_pair(SIZE, 0.0, math.pi / 2, generator=gen)
    assert img.shape == mask.shape == (SIZE, SIZE)
    assert 0.4 < mask.mean() < 0.6
    assert _ang_dist(_peak_orientation(img * (1 - mask)), 0.0) < 0.35
    assert _ang_dist(_peak_orientation(img * mask), math.pi / 2) < 0.35


@pytest.mark.parametrize("k", [-0.1, 0.0, 0.1])
def test_curvature_fragment_passes_center(k):
    img = curvature_fragment(SIZE, k)
    c = SIZE // 2
    assert img[c - 1 : c + 1, c - 1 : c + 1].max() > 0.5
    assert img.min() >= -1 - 1e-6
    # sign of curvature flips which side the arc bends toward
    if k != 0:
        top, bottom = img[: c - 8].sum(), img[c + 8 :].sum()
        assert (bottom > top) == (k > 0)


def test_identity_stack_end_to_end():
    x = torch.randn(4, 1, SIZE, SIZE)
    stack = build_stack([{"name": "V1", "kind": "identity"}, {"name": "V2", "kind": "identity"}])
    assert torch.equal(stack(x), x)
    assert IdentityStage("V1").out_channels(3) == 3
    with pytest.raises(KeyError):
        build_stack([{"name": "V1", "kind": "nope"}])


def test_viz_returns_figures():
    x = torch.randn(5, 1, 16, 16)
    for fig in (filter_atlas(x), fantasy_grid(x[:, 0]), sheet_map(torch.rand(8, 8), torch.rand(8, 8))):
        assert fig.axes
