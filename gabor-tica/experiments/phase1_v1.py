"""Phase 1 exit check: developmental V1 (fixed quadrature Gabors, energy,
divisive normalization, log).

Checks (plan Phase 1 "Validate"):
  1. orientation tuning: every unit peaks at its nominal orientation
  2. spatial-frequency tuning: every filter peaks near its nominal frequency
     (the normalization-induced shift is reported as a metric)
  3. complex-cell phase invariance
  4. contrast invariance: normalized tuning keeps its shape across contrast,
     while normalization compresses the contrast response
  5. heavy-tailed raw energy on natural images is ~Gaussian after the log

    python experiments/phase1_v1.py [--config configs/phase1.yaml]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from gtv.config import load_config  # noqa: E402
from gtv.data import REPO_PHOTOS, load_grayscale, random_patches  # noqa: E402
from gtv.probes import (  # noqa: E402
    excess_kurtosis,
    grating_responses,
    half_width_half_height,
    phase_modulation,
    skewness,
)
from gtv.stages import build_stage  # noqa: E402
from gtv.viz import filter_atlas  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INK = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#e4e3dc"
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal, light -> dark
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def ang_dist(a, b):
    d = (a - b) % math.pi
    return torch.minimum(d, math.pi - d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase1.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    v1 = build_stage(cfg["stages"][0])
    bank = v1.bank
    L, J = bank.n_orientations, bank.n_scales
    size = cfg["image_size"]
    t = cfg["tuning"]
    normalized = lambda x: v1.parts(x)["normalized"]  # noqa: E731
    raw_energy = lambda x: v1.parts(x)["energy"]  # noqa: E731
    metrics: dict = {}
    checks: dict = {}

    with torch.no_grad():
        # ---- 1. orientation tuning at each scale's preferred frequency
        thetas = torch.arange(t["n_thetas"]) * math.pi / t["n_thetas"]
        ori = grating_responses(normalized, size, bank.freq[::L], thetas)  # (J, T, C)
        pref_err, hwhh = [], []
        for c in range(bank.n_filters):
            curve = ori[c // L, :, c]
            pref_err.append(float(ang_dist(thetas[curve.argmax()], bank.theta[c])))
            hwhh.append(math.degrees(half_width_half_height(thetas, curve, circular_period=math.pi)))
        metrics["orientation_pref_error_deg_max"] = math.degrees(max(pref_err))
        metrics["orientation_hwhh_deg_median"] = float(torch.tensor(hwhh).median())
        checks["orientation_peaks"] = metrics["orientation_pref_error_deg_max"] <= 180 / t["n_thetas"] + 1e-6

        # ---- 2. spatial-frequency tuning at each unit's preferred orientation
        freqs = torch.logspace(math.log10(0.01), math.log10(0.45), t["n_freqs"])
        sf = torch.stack(
            [grating_responses(normalized, size, freqs, bank.theta[j * L : j * L + 1])[:, 0, j * L] for j in range(J)]
        )  # (J, F) for the theta=0 unit at each scale
        # the check uses filter energy (the unit's own preference); normalization
        # can shift the edge scales because the pool has no filters beyond the bank
        sf_ratio = {"energy": [], "normalized": []}
        for c in range(bank.n_filters):
            for name, fn in (("energy", raw_energy), ("normalized", normalized)):
                curve = grating_responses(fn, size, freqs, bank.theta[c : c + 1], n_phases=2)[:, 0, c]
                sf_ratio[name].append(float(freqs[curve.argmax()] / bank.freq[c]))
        metrics["sf_peak_ratio_range"] = {k: [min(v), max(v)] for k, v in sf_ratio.items()}
        tol = t["sf_peak_tolerance"]
        checks["sf_peaks"] = all(abs(math.log2(r)) <= tol for r in sf_ratio["energy"])

        # ---- 3. phase invariance
        pm = torch.stack(
            [phase_modulation(normalized, size, float(bank.freq[j * L]), 0.0)[j * L] for j in range(J)]
        )
        metrics["phase_modulation_max"] = float(pm.max())
        checks["phase_invariant"] = metrics["phase_modulation_max"] < t["max_phase_modulation"]

        # ---- 4. contrast invariance (theta=0 unit at the middle scale)
        contrasts = t["contrasts"]
        j = J // 2
        c0 = j * L
        f0 = bank.freq[c0 : c0 + 1]
        curves = {
            name: torch.stack([grating_responses(fn, size, f0, thetas, contrast=k)[0, :, c0] for k in contrasts])
            for name, fn in (("energy", raw_energy), ("normalized", normalized))
        }
        shapes = curves["normalized"] / curves["normalized"].max(1, keepdim=True).values
        corr = min(float(torch.corrcoef(torch.stack([shapes[-1], s]))[0, 1]) for s in shapes)
        peak = {k: float(v[-1].max() / v[0].max()) for k, v in curves.items()}
        metrics["contrast_shape_corr_min"] = corr
        metrics["peak_ratio_max_vs_min_contrast"] = peak
        checks["contrast_invariant_shape"] = corr > 0.99
        checks["contrast_compressed"] = peak["normalized"] < 0.1 * peak["energy"]

        # ---- 5. natural-image statistics
        imgs = load_grayscale(cfg["natural"]["folder"] or REPO_PHOTOS, max_side=cfg["natural"]["max_side"])
        x = random_patches(imgs, cfg["natural"]["n_patches"], size, gen)
        parts = v1.parts(x)
        m = slice(size // 4, 3 * size // 4)  # interior, away from padding
        stats = {}
        for k in ("energy", "normalized", "log"):
            v = parts[k][:, :, m, m]
            sk = torch.tensor([skewness(v[:, c]) for c in range(v.shape[1])])
            ku = torch.tensor([excess_kurtosis(v[:, c]) for c in range(v.shape[1])])
            stats[k] = {"skew_median": float(sk.median()), "excess_kurtosis_median": float(ku.median())}
        stats["log"]["fraction_at_floor"] = float(
            (parts["log"][:, :, m, m] <= math.log(v1.log.eps) + 0.05).float().mean()
        )
        metrics["natural_stats"] = stats
        g = t["gaussian_tolerance"]
        checks["log_gaussian"] = abs(stats["log"]["skew_median"]) < g and abs(stats["log"]["excess_kurtosis_median"]) < g
        checks["raw_heavy_tailed"] = stats["energy"]["excess_kurtosis_median"] > 3

    # ---- figures
    k = bank.kernels(49)
    filter_atlas(k.real, ncols=L).savefig(out / "filter_atlas_even.png", dpi=120)
    filter_atlas(k.imag, ncols=L).savefig(out / "filter_atlas_odd.png", dpi=120)

    deg = torch.rad2deg(thetas)
    fig, axes = plt.subplots(1, J, figsize=(3.2 * J, 2.6), sharey=True)
    for jj, ax in enumerate(axes):
        for l in range(L):
            ax.plot(deg, ori[jj, :, jj * L + l], color=SERIES[0], linewidth=1.2)
        _style(ax, f"scale {jj}: {float(bank.freq[jj * L]):.3g} cyc/px", "grating orientation (deg)",
               "normalized response" if jj == 0 else "")
    fig.suptitle("Orientation tuning, one curve per orientation unit", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "orientation_tuning.png", dpi=120)

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for jj in range(J):
        ax.plot(freqs, sf[jj], color=SERIES[jj], linewidth=2, label=f"scale {jj} ({float(bank.freq[jj * L]):.3g})")
        ax.axvline(float(bank.freq[jj * L]), color=SERIES[jj], linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    _style(ax, "Spatial-frequency tuning (0° units)", "grating frequency (cycles/pixel)", "normalized response")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "sf_tuning.png", dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for ax, name, title in zip(axes, ("energy", "normalized"), ("Raw energy", "After divisive normalization")):
        for i, kk in enumerate(contrasts):
            ax.plot(deg, curves[name][i], color=BLUE_RAMP[i + len(BLUE_RAMP) - len(contrasts)], linewidth=2,
                    label=f"contrast {kk}")
        _style(ax, title, "grating orientation (deg)", "response")
    axes[0].legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.suptitle("Contrast: tuning shape holds, amplitude is compressed", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "contrast_invariance.png", dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ref = torch.linspace(-4, 8, 200)
    for ax, key, title in zip(axes, ("energy", "log"), ("Raw energy", "log(normalized energy)")):
        v = parts[key][:, :, m, m]
        z = ((v - v.mean((0, 2, 3), keepdim=True)) / v.std((0, 2, 3), keepdim=True)).flatten()
        ax.hist(z.clamp(-4, 8).numpy(), bins=120, density=True, color=SERIES[0], alpha=0.85)
        ax.plot(ref, torch.exp(-0.5 * ref**2) / math.sqrt(2 * math.pi), color=MUTED, linewidth=1.2, label="N(0,1)")
        s = stats[key]
        ax.text(0.97, 0.95, f"skew {s['skew_median']:.2f}\nexcess kurt {s['excess_kurtosis_median']:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK)
        _style(ax, title, "z-score per channel", "density")
    axes[1].legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper left")
    fig.suptitle("Natural-image responses (repo photos)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "natural_distributions.png", dpi=120)

    report = {"metrics": metrics, "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Phase 1 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
