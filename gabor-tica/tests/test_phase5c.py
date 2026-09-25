import math

import torch
import torch.nn.functional as F

from gtv.stages import GaborBank, HigherStage, LearnedHigherStage, SpatialGaborBank, energy
from gtv.stages.hierarchy import build_stack
from gtv.temporal import fit_stage_filters


def _smooth_maps(n, c, hw, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.avg_pool2d(torch.randn(n, c, hw * 4, hw * 4, generator=g), 4) * 2


def test_spatial_bank_reproduces_frequency_bank():
    x = _smooth_maps(4, 3, 16)
    ref = GaborBank(4, 1, 0.25)(x)
    out = SpatialGaborBank(3, 4, 0.25, kernel_size=15)(x)
    assert out.shape == ref.shape and out.is_complex()
    # complex responses (not only energies): the kernel flip makes correlation equal convolution
    for part in (torch.real, torch.imag):
        c = torch.corrcoef(torch.stack([part(ref).flatten(), part(out).flatten()]))[0, 1]
        assert c > 0.999
    e_ref, e_out = energy(ref), energy(out)
    assert (e_out - e_ref).norm() / e_ref.norm() < 0.02


def test_learned_stage_matches_fixed_stage_and_tether():
    x = _smooth_maps(20, 6, 16)
    kw = dict(in_channels=6, sheet=4, radius=1, first_budget="all")
    fixed, learned = HigherStage("V4", **kw), LearnedHigherStage("V4", **kw)
    with torch.no_grad():
        f0, f1 = fixed.features(x), learned.features(x)
    assert f0.shape == f1.shape
    assert torch.corrcoef(torch.stack([f0.flatten(), f1.flatten()]))[0, 1] > 0.999
    assert learned.bank.drift() == 0.0
    params = [n for n, p in learned.named_parameters()]
    assert params == ["bank.weight"]  # the only trainable tensor; the init copy is a buffer
    with torch.no_grad():
        learned.bank.weight.mul_(1.1)
    assert math.isclose(learned.bank.drift(), 0.1, rel_tol=1e-4)
    # state dict round trip keeps both the learned kernels and the Gabor reference
    other = LearnedHigherStage("V4", **kw)
    other.load_state_dict(learned.state_dict())
    assert torch.equal(other.bank.weight, learned.bank.weight)
    assert torch.equal(other.bank.weight_init, learned.bank.weight_init)


def test_build_stack_selects_learned_filters():
    cfg = {"V4": {"sheet": 4, "radius": 1, "skip": None, "first_budget": "all", "learned_filters": True,
                  "kernel_size": 9},
           "PIT": {"sheet": 5, "radius": 1, "skip": None, "first_budget": "all"}}
    st = build_stack(cfg, {"V1": 8, "V2": 6})
    assert isinstance(st["V4"], LearnedHigherStage) and st["V4"].bank.radius == 4
    assert type(st["PIT"]) is HigherStage


def _persistent_pairs(n=48, c=2, hw=16, seed=0):
    """Frame pairs whose persistent content is an oblique grating (22.5 deg,
    between the bank's orientations) with a new phase each frame, and whose
    remaining content (smooth noise) is redrawn each frame."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(hw).float(), torch.arange(hw).float(), indexing="ij")
    th = math.pi / 8
    u = xx * math.cos(th) + yy * math.sin(th)
    amp = torch.rand(n, c, 1, 1, generator=g) * 2  # persistent: same amplitude in both frames

    def frame():
        ph = torch.rand(n, c, 1, 1, generator=g) * 2 * math.pi
        noise = F.avg_pool2d(torch.randn(n, c, hw * 2, hw * 2, generator=g), 2) * 1.5
        return amp * torch.cos(2 * math.pi * 0.25 * u + ph) + noise

    return frame(), frame()


def _frame_energy_corr(stage, a, b):
    """Mean over second-order channels of the across-sample correlation of the
    stage's log energies in the two frames."""
    with torch.no_grad():
        fa = stage.features(a)[:, stage.second_order_slice].mean((2, 3))
        fb = stage.features(b)[:, stage.second_order_slice].mean((2, 3))
    fa, fb = fa - fa.mean(0), fb - fb.mean(0)
    return float(((fa * fb).sum(0) / (fa.norm(dim=0) * fb.norm(dim=0) + 1e-8)).mean())


def test_temporal_learning_moves_filters_toward_persistent_content():
    torch.manual_seed(0)
    a, b = _persistent_pairs()
    kw = dict(in_channels=2, sheet=3, radius=1, first_budget="all", kernel_size=9)
    fit = dict(n_steps=150, lr=0.03, batch_size=16, refit_every=50, tica_kw={"n_iter": 40, "polish_iter": 10})
    init = LearnedHigherStage("V4", **kw)
    tethered, free = LearnedHigherStage("V4", **kw), LearnedHigherStage("V4", **kw)
    fit_stage_filters(tethered, a, b, temporal_weight=1.0, tether=100.0, **{**fit, "n_steps": 50})
    fit_stage_filters(free, a, b, temporal_weight=1.0, tether=0.0, **fit)
    assert tethered.bank.drift() < 0.2 * free.bank.drift()  # the tether holds the Gabors
    # ground truth: only the oblique grating's energy persists across frames, so
    # the learned energies should co-vary between the frames more than the
    # Gabors' do (init ~0.74, learned ~0.97 on held-out pairs)
    a2, b2 = _persistent_pairs(seed=1)
    assert _frame_energy_corr(free, a2, b2) > _frame_energy_corr(init, a2, b2) + 0.1
