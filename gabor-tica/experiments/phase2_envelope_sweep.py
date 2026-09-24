"""Envelope-frequency sweep for the second-order texture probe.

Phase 2 tested contrast-modulated textures at one envelope frequency that
happened to equal one of Design B's fixed second-order scales. This sweeps the
envelope frequency across ~3.5 octaves and compares:

  V1          V1 log energy alone (should stay near chance throughout)
  A           TICA learned on V1-map neighborhoods
  B           Phase 2 Design B, second-order scales 0.0625 / 0.03125
  B_shifted   same, scales moved half an octave up (0.088 / 0.044)
  B_wide      four second-order scales spanning the sweep

If B's advantage comes from the architecture, B should beat A across the
sweep; if it comes from the scale choice, B's curve should peak at its own
scales and B_shifted's peak should move with them.

    python experiments/phase2_envelope_sweep.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from gtv.config import load_config  # noqa: E402
from gtv.data import REPO_PHOTOS, load_grayscale, random_patches  # noqa: E402
from gtv.probes import linear_decode  # noqa: E402
from gtv.probes.sets import second_order_set  # noqa: E402
from gtv.stages import V2Stage, build_stage  # noqa: E402
from gtv.viz import fantasy_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
COLORS = {"A": "#eb6834", "B": "#2a78d6", "B_shifted": "#1baf7a", "B_wide": "#e87ba4", "V1": MUTED}
LABELS = {"V1": "V1 only", "A": "Design A", "B": "Design B", "B_shifted": "B, scales +½ octave", "B_wide": "B, 4 scales"}
SEEDS = (0, 1, 2)


def window_mean(y, w):
    lo = (y.shape[2] - w) // 2
    return y[:, :, lo : lo + w, lo : lo + w].mean((2, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase2_envelope_sweep.yaml"))
    args = ap.parse_args()
    sweep = load_config(args.config)
    cfg = load_config(ROOT / sweep["base_config"])
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    out = ROOT / sweep["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    size, p, vc = cfg["image_size"], cfg["probes"], cfg["v2"]

    v1 = build_stage({"name": "V1", **cfg["v1"]})
    common = dict(sheet=vc["sheet"], radius=vc["radius"], dim=vc["dim"], tica_mode=vc["tica_mode"])
    stages = {"A": V2Stage(design="A", **common, **vc["design_A"])}
    base_second = vc["design_B"]["second_order"]
    for name, freqs in sweep["b_variants"].items():
        stages[name] = V2Stage(
            design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"],
            second_order={**base_second, "freqs": freqs}, **common,
        )

    with torch.no_grad():
        nat = v1(random_patches(load_grayscale(REPO_PHOTOS), cfg["train"]["n_patches"], size, gen))
    for name, st in stages.items():
        st.fit(nat, n_iter=vc["n_iter"], seed=cfg["seed"])
        print(f"fitted {name}")

    w = p["window"]
    results = {k: [] for k in ["V1", *stages]}
    examples = []
    for f in sweep["envelope_freqs"]:
        x, y = second_order_set({**p, "cm_envelope_freq": f}, size, gen)
        examples.append(x[0])
        with torch.no_grad():
            m = v1(x)
            feats = {"V1": window_mean(m, 2 * w)}
            for name, st in stages.items():
                s = st(m)
                feats[name] = window_mean(torch.cat([s, st.pooled_energy(s)], 1), w)
        for k, v in feats.items():
            results[k].append(sum(linear_decode(v, y, seed=s) for s in SEEDS) / len(SEEDS))
        print(f"envelope {f:.4f}: " + "  ".join(f"{k} {results[k][-1]:.2f}" for k in results))

    freqs = sweep["envelope_freqs"]
    summary = {}
    for k in stages:
        diff = [b - a for b, a in zip(results[k], results["A"])]
        summary[k] = {
            "mean_acc": sum(results[k]) / len(freqs),
            "peak_envelope_freq": freqs[max(range(len(freqs)), key=lambda i: results[k][i])],
            "min_minus_A": min(diff),
            "n_freqs_beating_A_by_0.02": sum(d > 0.02 for d in diff),
            "n_freqs_trailing_A_by_0.02": sum(d < -0.02 for d in diff),
        }
    summary["V1"] = {"mean_acc": sum(results["V1"]) / len(freqs)}
    report = {"envelope_freqs": freqs, "accuracy": results, "summary": summary,
              "b_scales": sweep["b_variants"]}
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # ---- figure
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k in results:
        ls = ":" if k == "V1" else "-"
        ax.plot(freqs, results[k], ls, marker="o", markersize=5, linewidth=2, color=COLORS[k], label=LABELS[k])
    for k in ("B", "B_shifted"):
        for f in sweep["b_variants"][k]:
            ax.axvline(f, color=COLORS[k], linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(0.25, color=MUTED, linewidth=1, linestyle=":")
    ax.text(freqs[-1], 0.26, "chance", ha="right", va="bottom", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks(freqs, [f"{f:.3g}" for f in freqs])
    ax.minorticks_off()
    ax.set_ylim(0, 1.05)
    ax.set_title("Second-order texture decoding vs. envelope frequency\n(dashed: fixed second-order scales of B and B-shifted)",
                 fontsize=9, color=INK, loc="left")
    ax.set_xlabel("envelope frequency (cycles / pixel)", fontsize=8, color=MUTED)
    ax.set_ylabel("held-out accuracy (4-way)", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.tight_layout()
    fig.savefig(out / "envelope_sweep.png", dpi=120)
    fantasy_grid(torch.stack(examples), ncols=len(freqs), titles=[f"{f:.3g}" for f in freqs]).savefig(
        out / "envelope_examples.png", dpi=120)

    print(json.dumps(summary, indent=2))
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
