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


# ---- Phase 5d: Gabor mixing across sheet neighbors

def test_gabor_mix_identity_equals_fixed_bank_and_neighbors():
    from gtv.stages import GaborMixBank

    x = _smooth_maps(3, 16, 16)  # 16 channels = a 4 x 4 input sheet
    bank = GaborMixBank(16, 4, 0.25, mix_radius=1, offset=2)
    assert torch.equal(bank(x), GaborBank(4, 1, 0.25)(x))
    assert bank.nbr.shape == (16, 9) and (bank.nbr[:, 0] == torch.arange(16)).all()
    assert set(bank.nbr[0].tolist()) == {0, 1, 3, 4, 5, 7, 12, 13, 15}  # 3 x 3 on the 4 x 4 torus
    assert GaborMixBank(16, 4, 0.25, mix_radius=0, offset=0).weight.shape == (16, 4, 1, 4, 1, 2)


def test_gabor_mix_routes_neighbor_channels_offsets_and_phase():
    from gtv.stages import GaborMixBank

    x = _smooth_maps(2, 16, 16)
    bank = GaborMixBank(16, 4, 0.25, mix_radius=1, offset=2)
    ref = GaborBank(4, 1, 0.25)(x).view(2, 16, 4, 16, 16)
    j = int(bank.nbr[5, 3])
    with torch.no_grad():
        bank.weight.zero_()
        bank.weight[5, 1, 3, 2, 0, 1] = 1.0  # output (5, 1) = i * (channel j, orientation 2)
        bank.weight[6, 0, 0, 0, 3, 0] = 1.0  # output (6, 0) = own orientation 0, shifted by offsets[3]
    y = bank(x).view(2, 16, 4, 16, 16)
    assert torch.allclose(y[:, 5, 1], 1j * ref[:, j, 2], atol=1e-6)
    dy, dx = bank.offsets[3]
    assert dy == 2 and dx == 0  # content moves down 2 cells, zeros enter at the top
    assert torch.allclose(y[:, 6, 0, 2:], ref[:, 6, 0, :-2], atol=1e-6)
    assert y[:, 6, 0, :2].abs().max() == 0
    share = bank.offdiagonal_share()
    assert 0 < share["other_channels"] < 1 and 0 < share["shifted_positions"] < 1


def test_ngd_optimizer_trains_mixing_and_tether_limits_it():
    torch.manual_seed(0)
    a, b = _persistent_pairs(n=32, c=4)
    kw = dict(in_channels=4, sheet=4, radius=1, first_budget="all", bank="mix", mix_radius=1, offset=2)
    fit = dict(n_steps=40, lr=0.01, batch_size=16, refit_every=20, optimizer="ngd",
               tica_kw={"n_iter": 30, "polish_iter": 5})
    loose, tight = LearnedHigherStage("V4", **kw), LearnedHigherStage("V4", **kw)
    fit_stage_filters(loose, a, b, tether=0.0, **fit)
    fit_stage_filters(tight, a, b, tether=1000.0, **fit)
    assert 0 < tight.bank.drift() < loose.bank.drift()
    assert torch.isfinite(loose.bank.weight).all()


def test_sequence_bubbles_generalizes_pairs_and_rewards_persistence():
    from gtv.stages.tica import torus_neighborhood
    from gtv.temporal import bubbles_loss, bubbles_seq_loss

    torch.manual_seed(0)
    h = torus_neighborhood(3, 3, 1)
    a, b = torch.randn(50, 9), torch.randn(50, 9)
    assert torch.allclose(bubbles_seq_loss([a, b], h, 0.7), bubbles_loss(a, b, h, 0.7))
    # energy that stays in the same units over 4 frames costs less than energy that hops
    s = torch.randn(50, 9)
    stay = [s * torch.randn(50, 1).abs() for _ in range(4)]
    hop = [s[:, torch.randperm(9)] * torch.randn(50, 1).abs() for _ in range(4)]
    far = torch.zeros(9, 9) + torch.eye(9)  # unit-only pools: hopping cannot hide in a shared pool
    assert bubbles_seq_loss(stay, far, 1.0) < bubbles_seq_loss(hop, far, 1.0)
    # sequences through the trainer: (N, T, C, H, W) with x_t1 None
    st = LearnedHigherStage("V4", in_channels=4, sheet=4, radius=1, first_budget="all", bank="mix", mix_radius=1)
    seq = _smooth_maps(12, 4, 16).unsqueeze(1).repeat(1, 3, 1, 1, 1)
    hist = fit_stage_filters(st, seq, None, n_steps=3, batch_size=4, refit_every=3, refit_n=6,
                             optimizer="ngd", lr=0.01, tica_kw={"n_iter": 5, "polish_iter": 1})
    assert len(hist["bubbles"]) == 3 and st.bank.drift() > 0


def test_build_stack_mixing_stages_and_state_round_trip():
    cfg = {"V4": {"sheet": 4, "radius": 1, "skip": None, "first_budget": "all", "bank": "mix", "mix_radius": 1,
                  "offset": 2},
           "PIT": {"sheet": 5, "radius": 1, "skip": None, "first_budget": "all", "bank": "mix", "offset": 1}}
    st = build_stack(cfg, {"V1": 8, "V2": 9})
    assert st["V4"].bank.nbr.shape == (9, 9) and st["PIT"].bank.offsets[1] == (0, 1)
    with torch.no_grad():
        st["V4"].bank.weight.add_(0.01)
    other = build_stack(cfg, {"V1": 8, "V2": 9})
    other["V4"].load_state_dict(st["V4"].state_dict())
    assert torch.equal(other["V4"].bank.weight, st["V4"].bank.weight)
