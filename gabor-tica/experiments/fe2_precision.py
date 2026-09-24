"""Section 7, step FE-2: learn a precision per V1 channel.

For V1 with and without divisive normalization, compare FE-1's fixed
precision with learned per-channel precision (held at FE-1's value for the
first steps, then freed and bounded). Reports what the model learns to trust,
whether that matches the maximum-likelihood precision, whether it fixes FE-1's
second-order readout, and how divisive normalization changes the precision
profile across channels.

    python experiments/fe2_precision.py [--config configs/fe2.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from fe1_free_energy import topography, window_mean  # noqa: E402

from gtv.config import load_config  # noqa: E402
from gtv.data import REPO_PHOTOS, load_grayscale, random_patches  # noqa: E402
from gtv.generative import (  # noqa: E402
    AmortizedEncoder,
    ConvFactorAnalysis,
    TICAPrior,
    active_units,
    encoder_r2,
    fit_free_energy,
    residual_log_precision,
)
from gtv.probes import linear_decode  # noqa: E402
from gtv.probes.sets import second_order_set  # noqa: E402
from gtv.stages import V2Stage, build_stage  # noqa: E402

INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEEDS = (0, 1, 2)


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "fe2.yaml"))
    args = ap.parse_args()
    c2 = load_config(args.config)
    c1 = load_config(ROOT / c2["base_config"])
    c3 = load_config(ROOT / c1["base_config"])
    cfg = load_config(ROOT / c3["base_config"])
    vc, d, t, pc = cfg["v2"], c3["data"], c1["train"], c2["precision"]
    out = ROOT / c2["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["image_size"]
    photos = load_grayscale(REPO_PHOTOS)
    results, profiles = {}, {}

    for variant, normalize in c2["v1_variants"].items():
        gen = torch.Generator().manual_seed(cfg["seed"])
        v1 = build_stage({"name": "V1", **cfg["v1"], "normalize": normalize})
        L, J = v1.bank.n_orientations, v1.bank.n_scales
        with torch.no_grad():
            tr = v1(random_patches(photos[: d["train_photos"]], d["n_train"], size, gen))
            te = v1(random_patches(photos[d["train_photos"] :], d["n_test"], size, gen))
        v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"],
                     sheet=vc["sheet"], radius=vc["radius"], dim=vc["dim"], tica_mode=vc["tica_mode"],
                     **vc["design_B"])
        v2.fit(tr[: d["n_v2_fit"]], n_iter=vc["n_iter"], seed=cfg["seed"])
        prior = TICAPrior(v2.tica.h, v2.tica.eps)
        n_lat, rows = v2.tica.n_units, v2.tica.rows
        probes = {}
        for f in c1["probes"]["envelope_freqs"]:
            x, y = second_order_set({**cfg["probes"], "cm_envelope_freq": f}, size, gen)
            with torch.no_grad():
                probes[f] = (v1(x), y)

        for mode in ("fixed", "learned"):
            g = torch.Generator().manual_seed(cfg["seed"])
            enc = AmortizedEncoder(v2, init="tica")
            dec = ConvFactorAnalysis(n_lat, tr.shape[1], c3["decoder"]["kernel"])
            dec.fit_standardizer(tr)
            hist = fit_free_energy(
                tr, enc, dec, prior, torch.tensor(pc["init_log"]), n_iter=t["n_iter"], batch=t["batch"], lr=t["lr"],
                warmup=t["warmup"], arrange_every=t["arrange_every"], learn_precision=(mode == "learned"),
                free_precision_after=pc["free_after"], log_precision_bounds=tuple(pc["bounds"]),
                lr_precision=pc["lr"], generator=g, log_every=50)
            act = active_units(enc, te)
            with torch.no_grad():
                mu, _ = enc(te)
            acc = {}
            for f, (m, y) in probes.items():
                with torch.no_grad():
                    mp, _ = enc(m)
                    pooled = torch.sqrt(torch.einsum("ij,bjhw->bihw", v2.tica.h, mp.pow(2)) + v2.tica.eps)
                feat = window_mean(torch.cat([mp, pooled], 1))
                acc[str(f)] = sum(linear_decode(feat, y, seed=s) for s in SEEDS) / len(SEEDS)
            lp = hist["log_precision"].expand(tr.shape[1]).clone()
            mle = residual_log_precision(enc, dec, tr)
            key = f"{variant}/{mode}"
            results[key] = {
                "test_r2": encoder_r2(enc, dec, te),
                "active_units": int(act.sum()),
                "topography_active": topography(mu, act, rows),
                "second_order_acc": acc,
                "log_precision": lp.view(J, L).tolist(),
                "ml_log_precision": mle.view(J, L).tolist(),
                "log_precision_by_scale": lp.view(J, L).mean(1).tolist(),
                "log_precision_spread_across_channels": float(lp.std()),
                "mle_gap_median": float((lp - mle).abs().median()),
                "log_precision_mean_history": hist["log_precision_mean"],
                "history_iter": hist["iter"],
                "F_final": hist["F_per_element"][-1],
            }
            profiles[key] = (lp.view(J, L), mle.view(J, L))
            print(f"{key}: R2 {results[key]['test_r2']:.3f} active {results[key]['active_units']} "
                  f"2nd-order {acc} logpi by scale {[round(v, 2) for v in results[key]['log_precision_by_scale']]}",
                  flush=True)
        # interaction with normalization: spread of the ML precision across channels
        # (defined for fixed-precision runs too)
        for mode in ("fixed", "learned"):
            key = f"{variant}/{mode}"
            results[key]["ml_log_precision_spread"] = float(profiles[key][1].std())

    # ---- checks (declared in the config before running)
    ch = c2["checks"]
    learned = [k for k in results if k.endswith("/learned")]
    lo, hi = c2["precision"]["bounds"]
    checks = {
        "no_runaway_precision": all(lo + 0.05 < min(min(r) for r in results[k]["log_precision"]) and
                                    max(max(r) for r in results[k]["log_precision"]) < hi - 0.05 for k in learned),
        "learned_matches_ml_precision": all(results[k]["mle_gap_median"] < ch["max_mle_gap"] for k in learned),
        "second_order_readout_fixed": results["normalized/learned"]["second_order_acc"]["0.0625"] >= ch["min_second_order_acc"],
        "reconstruction_kept": all(results[f"{v}/learned"]["test_r2"] >= results[f"{v}/fixed"]["test_r2"] - ch["max_r2_below_fe1"]
                                   for v in c2["v1_variants"]),
    }
    findings = {
        "precision_spread_normalized": results["normalized/learned"]["log_precision_spread_across_channels"],
        "precision_spread_unnormalized": results["unnormalized/learned"]["log_precision_spread_across_channels"],
        "normalization_equalizes_precision": results["normalized/learned"]["log_precision_spread_across_channels"]
        < results["unnormalized/learned"]["log_precision_spread_across_channels"],
    }

    # ---- figures
    keys = [k for k in results if k.endswith("/learned")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 2.6))
    vmin = min(profiles[k][0].min() for k in keys)
    vmax = max(profiles[k][0].max() for k in keys)
    for ax, k in zip(axes, keys):
        im = ax.imshow(profiles[k][0].numpy(), cmap="Blues", vmin=float(vmin), vmax=float(vmax), aspect="auto")
        prof = profiles[k][0]
        for j in range(prof.shape[0]):
            for l in range(prof.shape[1]):
                v = float(prof[j, l])
                ax.text(l, j, f"{v:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if v > (float(vmin) + float(vmax)) / 2 else INK)
        ax.set_xticks(range(profiles[k][0].shape[1]), [f"{int(180 * l / profiles[k][0].shape[1])}°" for l in range(profiles[k][0].shape[1])], fontsize=6)
        ax.set_yticks(range(profiles[k][0].shape[0]), ["fine", "mid", "coarse"][: profiles[k][0].shape[0]], fontsize=7)
        ax.set_title(f"Learned log precision, V1 {k.split('/')[0]}", fontsize=9, color=INK, loc="left")
    fig.colorbar(im, ax=axes, fraction=0.02, label="log π")
    fig.savefig(out / "fe2_precision_profiles.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for i, k in enumerate(keys):
        ax.plot(results[k]["history_iter"], results[k]["log_precision_mean_history"], color=SERIES[i], linewidth=2,
                label=f"V1 {k.split('/')[0]}")
    ax.axvline(pc["free_after"], color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(pc["free_after"], ax.get_ylim()[0], " precision freed", fontsize=7, color=MUTED, va="bottom")
    _style(ax, "Mean learned log precision during training", "iteration", "mean log π")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "fe2_precision_history.png", dpi=120)

    report = {"results": results, "checks": checks, "findings": findings, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    brief = {k: {kk: v[kk] for kk in ("test_r2", "active_units", "topography_active", "second_order_acc",
                                        "log_precision_by_scale", "log_precision_spread_across_channels",
                                        "ml_log_precision_spread", "mle_gap_median")} for k, v in results.items()}
    print(json.dumps({"results": brief, "checks": checks, "findings": findings, "passed": report["passed"]}, indent=2))
    print(f"FE-2 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
