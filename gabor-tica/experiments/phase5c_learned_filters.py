"""Phase 5c: Gabor-initialized, learnable V4 filters trained by temporal coherence.

Phase 5b's upper stages add no invariance: their filters are fixed Gabors and
the only learned part (complete TICA) is a rotation. Here V4's second-order
Gabors become learnable spatial kernels (``LearnedHigherStage``), initialized
from the fixed bank, and are trained on Fashion-MNIST frame pairs (an image and
a slightly shifted / rotated / rescaled copy) with the bubbles objective, a
whiteness term and a tether to the Gabor initialization
(``gtv.temporal.fit_stage_filters``). V1, V2 are the Phase 5b stack, loaded.

Step 1 (refactor check): with learning off, the spatial-kernel V4 must
reproduce the fixed V4 (matched readout within ``reproduce_tolerance``).
Step 2: sweep the tether for "still" (sparsity only, temporal weight 0) and
"bubbles" (temporal pooling) and measure V4 matched-readout tolerance and the
unit invariance index against the fixed V4.

    python experiments/phase5c_learned_filters.py [--config configs/phase5c.yaml] [--quick]
"""

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gtv.config import load_config  # noqa: E402
from gtv.data import CLASSES, affine, load_fashion_mnist  # noqa: E402
from gtv.probes.decode import fit_logistic  # noqa: E402
from gtv.stages import HigherStage, LearnedHigherStage, V2Stage, build_stage  # noqa: E402
from gtv.temporal import fit_stage_filters  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
MODE_COLORS = {"still": "#c3c2b7", "bubbles": "#2a78d6"}


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase5c.yaml"))
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a smoke run")
    args = ap.parse_args()
    c = load_config(args.config)
    c5 = load_config(ROOT / c["phase5_config"])
    cfg = load_config(ROOT / c5["base_config"])
    vc, d, tica_kw = cfg["v2"], dict(c5["data"]), dict(c5["tica"])
    tr, sweep, mr = dict(c["train"]), list(c["sweep"]), dict(c5["matched_readout"])
    if args.quick:
        d.update(n_fit=200, n_decode_test=200)
        tr.update(n_pairs=64, n_steps=6, refit_every=3)
        mr.update(n_train=400, pca_dims=32)
        tica_kw = {"n_iter": 10, "polish_iter": 2}
        sweep = sweep[:1]
    out = ROOT / c["output_dir"] / ("quick" if args.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    size, seed = cfg["image_size"], cfg["seed"]
    sc = c5["stages"]["V4"]
    ckpt_path = ROOT / c["stack_checkpoint"]
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} not found: run experiments/phase5_hierarchy.py --config configs/phase5b.yaml")

    # ---- V1, V2 and the fixed V4 of Phase 5b, loaded
    v1 = build_stage({"name": "V1", **cfg["v1"]})
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"], sheet=vc["sheet"],
                 radius=vc["radius"], dim=vc["dim"], tica_mode=vc["tica_mode"], **vc["design_B"])
    ck = torch.load(ckpt_path)
    v2.load_state_dict(ck["V2"])
    n_in = v2.tica.n_units
    stage_kw = {"first_budget": sc.get("first_budget")}
    fixed = HigherStage("V4", n_in, sc["sheet"], sc["radius"], **stage_kw)
    fixed.load_state_dict(ck["V4"])

    def v2_maps(x):
        with torch.no_grad():
            return torch.cat([v2(v1(xb)) for xb in x.split(128)])

    def run(stage, s2):
        with torch.no_grad():
            return torch.cat([stage(b) for b in s2.split(256)])

    # ---- data (same seeds as Phase 5b for fitting and decoding)
    s2_fit = v2_maps(load_fashion_mnist("train", d["n_fit"], size, seed=1)[0])
    x_big, y_big = load_fashion_mnist("train", mr["n_train"], size, seed=7)
    s2_big = v2_maps(x_big)
    del x_big
    x_te, y_te = load_fashion_mnist("test", d["n_decode_test"], size, seed=3)
    n = len(x_te)
    s2_te = {"original": v2_maps(x_te)}
    for tname, (sx, rot, scl) in c5["transforms"].items():
        with torch.no_grad():
            xt = affine(x_te, torch.tensor([[sx, 0.0]]).expand(n, 2), torch.full((n,), rot), torch.full((n,), scl))
        s2_te[tname] = v2_maps(xt)
    tnames = list(c5["transforms"])

    # training pairs: an image and a slightly transformed copy (a drift step)
    g = torch.Generator().manual_seed(tr["pair_seed"])
    x_p, _ = load_fashion_mnist("train", tr["n_pairs"], size, seed=tr["pair_seed"])
    npr = len(x_p)
    shift = (torch.rand(npr, 2, generator=g) * 2 - 1) * tr["max_shift"]
    rot = (torch.rand(npr, generator=g) * 2 - 1) * tr["max_rot_deg"]
    scale = torch.exp((torch.rand(npr, generator=g) * 2 - 1) * tr["max_log_scale"])
    with torch.no_grad():
        p_t, p_t1 = v2_maps(x_p), v2_maps(affine(x_p, shift, rot, scale))
    print(f"data: fit {tuple(s2_fit.shape)}, readout train {len(s2_big)}, test {n} x {len(s2_te)}, pairs {npr}",
          flush=True)

    # ---- evaluation: matched readout (as Phase 5b) and unit invariance index
    def readout(stage, s):
        return F.adaptive_avg_pool2d(torch.cat([s, stage.pooled_energy(s)], 1), 2).flatten(1)

    def matched(f_big, f_te):
        mu, sd = f_big.mean(0), f_big.std(0) + 1e-6
        zt = (f_big - mu) / sd
        _, _, vh = torch.linalg.svd(zt - zt.mean(0), full_matrices=False)
        proj = vh[: min(mr["pca_dims"], vh.shape[0])].T
        zp = zt @ proj
        m2, s2 = zp.mean(0), zp.std(0) + 1e-6
        with torch.enable_grad():
            model = fit_logistic((zp - m2) / s2, y_big, len(CLASSES), l2=1e-3)
        with torch.no_grad():
            return {k: float((model(((((f - mu) / sd) @ proj) - m2) / s2).argmax(1) == y_te).float().mean())
                    for k, f in f_te.items()}

    def invariance(pe):
        """Median over features of the across-image correlation between a
        2x2-pooled pooled-energy feature for originals and transformed images."""
        a = pe["original"]
        res = {}
        for t in tnames:
            b = pe[t]
            az, bz = a - a.mean(0), b - b.mean(0)
            r = (az * bz).sum(0) / (az.norm(dim=0) * bz.norm(dim=0) + 1e-8)
            res[t] = float(r.median())
        return res

    def evaluate(stage, label):
        if stage is v2:
            outs_big, outs_te = s2_big, s2_te
        else:
            outs_big, outs_te = run(stage, s2_big), {k: run(stage, v) for k, v in s2_te.items()}
        f_te = {k: readout(stage, v) for k, v in outs_te.items()}
        acc = matched(readout(stage, outs_big), f_te)
        pe = {k: F.adaptive_avg_pool2d(stage.pooled_energy(v), 2).flatten(1) for k, v in outs_te.items()}
        inv = invariance(pe)
        r = {"matched": acc, "mean_transformed": sum(acc[t] for t in tnames) / len(tnames),
             "invariance": inv, "mean_invariance": sum(inv.values()) / len(inv)}
        print(f"{label}: matched {({k: round(v, 3) for k, v in acc.items()})}; mean transformed "
              f"{r['mean_transformed']:.3f}; invariance {({k: round(v, 3) for k, v in inv.items()})} "
              f"(mean {r['mean_invariance']:.3f})", flush=True)
        return r

    results = {"V2": evaluate(v2, "V2"), "V4_fixed": evaluate(fixed, "V4 fixed (Phase 5b)")}

    # ---- step 1: learning off (Gabor-initialized spatial kernels), fitted as Phase 5b
    init = LearnedHigherStage("V4", n_in, sc["sheet"], sc["radius"], kernel_size=c["kernel_size"], **stage_kw)
    init.fit(s2_fit, border=sc["border"], seed=seed, **tica_kw)
    with torch.no_grad():
        e_fixed = fixed.features(s2_te["original"][:200])[:, fixed.second_order_slice]
        e_init = init.features(s2_te["original"][:200])[:, init.second_order_slice]
        feature_corr = float(torch.corrcoef(torch.stack([e_fixed.flatten(), e_init.flatten()]))[0, 1])
    print(f"learning off: second-order feature correlation with the fixed V4 {feature_corr:.5f}", flush=True)
    results["V4_init"] = evaluate(init, "V4 learning off")
    results["V4_init"]["feature_corr_with_fixed"] = feature_corr

    # ---- step 2: learned filters, tether sweep, still vs bubbles
    histories, stages_out, kernels = {}, {}, {}
    for v in sweep:
        label = f"{v['mode']}_tether{v['tether']:g}"
        st = copy.deepcopy(init)
        hist = fit_stage_filters(st, p_t, p_t1, temporal_weight=c["temporal_weights"][v["mode"]],
                                 tether=v["tether"], white_weight=tr["white_weight"], n_steps=tr["n_steps"],
                                 lr=tr["lr"], batch_size=tr["batch_size"], refit_every=tr["refit_every"],
                                 refit_n=tr["refit_n"], border=sc["border"], seed=seed, tica_kw=tica_kw)
        st.fit(s2_fit, border=sc["border"], seed=seed, **tica_kw)
        r = evaluate(st, f"V4 {label} (drift {hist['drift'][-1]:.3f})")
        r.update(mode=v["mode"], tether=v["tether"], drift=hist["drift"][-1],
                 final_losses={k: hist[k][-1] for k in ("bubbles", "white", "tether")})
        results[label], histories[label] = r, hist
        stages_out[label] = st.state_dict()
        kernels[label] = st.bank.weight.detach().clone()

    # ---- checks (declared before running)
    ch = c["checks"]
    base, off = results["V4_fixed"], results["V4_init"]
    reproduce = max(abs(off["matched"][k] - base["matched"][k]) for k in base["matched"])
    learned = {k: r for k, r in results.items() if "mode" in r}
    keep = {k: r for k, r in learned.items()
            if r["matched"]["original"] >= base["matched"]["original"] - ch["max_accuracy_drop"]}
    tol_gain = {k: r["mean_transformed"] - base["mean_transformed"] for k, r in learned.items()}
    inv_gain = {k: r["mean_invariance"] - base["mean_invariance"] for k, r in learned.items()}
    bubbles_kept = [k for k in keep if learned[k]["mode"] == "bubbles"]
    best = max(bubbles_kept, key=lambda k: tol_gain[k]) if bubbles_kept else None
    checks = {
        "learning_off_reproduces_fixed_v4": reproduce <= ch["reproduce_tolerance"],
        "readout_tolerance_gain": bool(best) and tol_gain[best] >= ch["min_tolerance_gain"],
        "invariance_index_gain": any(inv_gain[k] >= ch["min_invariance_gain"] for k in bubbles_kept),
    }
    if best:
        still = f"still_tether{learned[best]['tether']:g}"
        if still in learned:
            checks["gain_is_temporal"] = tol_gain[best] - tol_gain[still] >= ch["min_temporal_margin"]
    summary = {"reproduce_max_diff": reproduce, "tolerance_gain": tol_gain, "invariance_gain": inv_gain,
               "best_bubbles_variant": best}
    print(json.dumps(summary, indent=2), flush=True)

    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    labels = list(learned)
    xs = range(len(labels))
    cols = [MODE_COLORS[learned[k]["mode"]] for k in labels]
    short = [f"{learned[k]['mode']}\nτ={learned[k]['tether']:g}" for k in labels]
    for ax, gains, title in ((axes[0], tol_gain, "Matched-readout accuracy on transformed images vs fixed V4"),
                             (axes[1], inv_gain, "Unit invariance index vs fixed V4")):
        ax.bar(xs, [gains[k] for k in labels], color=cols, width=0.65)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_xticks(list(xs), short)
        _style(ax, title, "", "change")
    axes[0].axhline(ch["min_tolerance_gain"], color=MUTED, linestyle=":", linewidth=1)
    axes[1].axhline(ch["min_invariance_gain"], color=MUTED, linestyle=":", linewidth=1)
    ax = axes[2]
    for k in labels:
        ax.plot(histories[k]["drift"], color=MODE_COLORS[learned[k]["mode"]],
                linewidth=1.5 if learned[k]["mode"] == "bubbles" else 1)
        ax.text(len(histories[k]["drift"]) - 1, histories[k]["drift"][-1], f" τ={learned[k]['tether']:g}",
                fontsize=6, color=MUTED, va="center")
    for mode, col in MODE_COLORS.items():
        ax.plot([], [], color=col, label=mode)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    _style(ax, "Filter drift from the Gabor initialization", "step", "‖K − K₀‖ / ‖K₀‖")
    fig.tight_layout()
    fig.savefig(out / "phase5c_sweep.png", dpi=120)

    if best:
        w0, w1 = init.bank.weight_init, kernels[best]
        nf = init.bank.n_filters
        per_ch = (w1 - w0).view(n_in, -1).pow(2).sum(1)
        top = per_ch.argsort(descending=True)[:3].tolist()
        fig, axes = plt.subplots(len(top) * 2, nf, figsize=(nf * 1.2, len(top) * 2.5))
        for i, ch_i in enumerate(top):
            for j in range(nf):
                for row, w in ((2 * i, w0), (2 * i + 1, w1)):
                    k = w[(ch_i * nf + j) * 2, 0]
                    lim = float(w0[(ch_i * nf + j) * 2, 0].abs().max())
                    axes[row, j].imshow(k.numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
                    axes[row, j].set_xticks([])
                    axes[row, j].set_yticks([])
                axes[2 * i, 0].set_ylabel(f"ch {ch_i}\nGabor", fontsize=7, color=MUTED)
                axes[2 * i + 1, 0].set_ylabel("learned", fontsize=7, color=MUTED)
        fig.suptitle(f"Even kernels of the 3 most-changed input channels ({best})", fontsize=9, color=INK)
        fig.tight_layout()
        fig.savefig(out / "phase5c_kernels.png", dpi=120)

    report = {"results": results, "summary": summary, "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    torch.save(stages_out, out / "phase5c_v4_variants.pt")
    print(json.dumps({"checks": checks}, indent=2))
    print(f"Phase 5c {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
