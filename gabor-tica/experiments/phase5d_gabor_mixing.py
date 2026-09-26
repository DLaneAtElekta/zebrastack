"""Phase 5d: V4 filters as learned combinations of Gabors across V2 sheet neighbors.

Phase 5c let each V4 Gabor kernel change freely but stay on its own V2
channel; it gave no reliable invariance, and Adam put pixel-level noise into
the kernels. Here V4's second-order filters are ``GaborMixBank``s: complex-
weighted sums of the fixed Gabor responses (4 orientations x 5 spatial
offsets) of the V2 channels within ``mix_radius`` of the filter's own channel
on the V2 TICA sheet. The identity start equals the fixed V4 exactly, the
basis is smooth by construction, and radius 0 (own channel only: orientation
and offset mixing) is the per-channel control for radius 1 (3 x 3 sheet
neighbors: cross-feature combinations).

Each trained variant is evaluated after ``len(refit_seeds)`` whitening + TICA
refits, and so is the learning-off stage (the baseline), so every gain is
compared with the refit noise (mean +- standard error over seeds).

Phase 5e (``configs/phase5e.yaml``) uses the same script with training
sequences whose transforms span the test range (``ranges``, per-variant
``range`` and ``seq_len``: frame k of T is the original moved by k / (T - 1)
of a random end transform) and with extrapolation transforms beyond it
(``extra_transforms``, reported separately from the tested mean).

    python experiments/phase5d_gabor_mixing.py [--config configs/phase5d.yaml] [--quick]
"""

import argparse
import copy
import json
import math
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


def _mean_se(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return float(t.mean()), (float(t.std()) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase5d.yaml"))
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a smoke run")
    args = ap.parse_args()
    c = load_config(args.config)
    c5 = load_config(ROOT / c["phase5_config"])
    cfg = load_config(ROOT / c5["base_config"])
    vc, d, tica_kw = cfg["v2"], dict(c5["data"]), dict(c5["tica"])
    tr, sweep, mr, seeds = dict(c["train"]), list(c["sweep"]), dict(c5["matched_readout"]), list(c["refit_seeds"])
    if args.quick:
        d.update(n_fit=200, n_decode_test=200)
        tr.update(n_pairs=64, n_steps=6, refit_every=3, refit_n=32)
        mr.update(n_train=400, pca_dims=32)
        tica_kw = {"n_iter": 10, "polish_iter": 2}
        sweep, seeds = sweep[:2], seeds[:2]
    out = ROOT / c["output_dir"] / ("quick" if args.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["image_size"]
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

    # ---- data (same seeds as Phase 5b / 5c)
    s2_fit = v2_maps(load_fashion_mnist("train", d["n_fit"], size, seed=1)[0])
    x_big, y_big = load_fashion_mnist("train", mr["n_train"], size, seed=7)
    s2_big = v2_maps(x_big)
    del x_big
    x_te, y_te = load_fashion_mnist("test", d["n_decode_test"], size, seed=3)
    n = len(x_te)
    s2_te = {"original": v2_maps(x_te)}
    extra = dict(c.get("extra_transforms") or {})
    for tname, (sx, rot, scl) in {**c5["transforms"], **extra}.items():
        with torch.no_grad():
            xt = affine(x_te, torch.tensor([[sx, 0.0]]).expand(n, 2), torch.full((n,), rot), torch.full((n,), scl))
        s2_te[tname] = v2_maps(xt)
    tnames, enames = list(c5["transforms"]), list(extra)
    x_p, _ = load_fashion_mnist("train", tr["n_pairs"], size, seed=tr["pair_seed"])
    npr = len(x_p)
    ranges = c.get("ranges") or {"default": {k: tr[k] for k in ("max_shift", "max_rot_deg", "max_log_scale")}}
    seq_cache = {}

    def sequences(range_name, t_len):
        """(N, T, C, H, W) V2 maps: frame k is the image moved by k / (T - 1) of a
        random end transform drawn from the range (T = 2: an image and its
        transformed copy, as in Phase 5c/5d). The same draws for every T."""
        key = (range_name, t_len)
        if key not in seq_cache:
            rg = ranges[range_name]
            g = torch.Generator().manual_seed(tr["pair_seed"])
            shift = (torch.rand(npr, 2, generator=g) * 2 - 1) * rg["max_shift"]
            rot = (torch.rand(npr, generator=g) * 2 - 1) * rg["max_rot_deg"]
            log_s = (torch.rand(npr, generator=g) * 2 - 1) * rg["max_log_scale"]
            frames = [v2_maps(x_p)]
            for k in range(1, t_len):
                a = k / (t_len - 1)
                with torch.no_grad():
                    frames.append(v2_maps(affine(x_p, a * shift, a * rot, torch.exp(a * log_s))))
            seq_cache[key] = torch.stack(frames, 1)
        return seq_cache[key]

    print(f"data: fit {tuple(s2_fit.shape)}, readout train {len(s2_big)}, test {n} x {len(s2_te)}, sequences {npr}",
          flush=True)

    # ---- evaluation (as Phase 5c)
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
        a = pe["original"]
        res = {}
        for t in tnames + enames:
            b = pe[t]
            az, bz = a - a.mean(0), b - b.mean(0)
            res[t] = float(((az * bz).sum(0) / (az.norm(dim=0) * bz.norm(dim=0) + 1e-8)).median())
        return res

    def evaluate_once(stage):
        outs_big, outs_te = run(stage, s2_big), {k: run(stage, v) for k, v in s2_te.items()}
        acc = matched(readout(stage, outs_big), {k: readout(stage, v) for k, v in outs_te.items()})
        inv = invariance({k: F.adaptive_avg_pool2d(stage.pooled_energy(v), 2).flatten(1) for k, v in outs_te.items()})
        r = {"matched": acc, "mean_transformed": sum(acc[t] for t in tnames) / len(tnames),
             "invariance": inv, "mean_invariance": sum(inv[t] for t in tnames) / len(tnames)}
        if enames:
            r["mean_extrapolation"] = sum(acc[t] for t in enames) / len(enames)
            r["mean_extrapolation_invariance"] = sum(inv[t] for t in enames) / len(enames)
        return r

    def evaluate_seeds(stage, label):
        """Refit whitening + TICA with each seed (filters unchanged) and evaluate."""
        per = []
        for s in seeds:
            stage.fit(s2_fit, border=sc["border"], seed=s, **tica_kw)
            per.append(evaluate_once(stage))
        r = {"per_seed": per}
        for key in ("mean_transformed", "mean_invariance", "mean_extrapolation", "mean_extrapolation_invariance"):
            if key in per[0]:
                r[key], r[key + "_se"] = _mean_se([p[key] for p in per])
        for k in ["original"] + tnames + enames:
            r[k], r[k + "_se"] = _mean_se([p["matched"][k] for p in per])
        print(f"{label}: original {r['original']:.3f}±{r['original_se']:.3f}; "
              + "; ".join(f"{t} {r[t]:.3f}" for t in tnames)
              + f"; mean transformed {r['mean_transformed']:.3f}±{r['mean_transformed_se']:.3f}; "
              f"invariance {r['mean_invariance']:.3f}±{r['mean_invariance_se']:.3f}"
              + (f"; extrapolation {r['mean_extrapolation']:.3f}±{r['mean_extrapolation_se']:.3f} "
                 f"(invariance {r['mean_extrapolation_invariance']:.3f})" if enames else ""), flush=True)
        return r

    ref = evaluate_once(fixed)
    print(f"V4 fixed (Phase 5b checkpoint): mean transformed {ref['mean_transformed']:.3f}, "
          f"invariance {ref['mean_invariance']:.3f}", flush=True)
    results = {"V4_fixed_checkpoint": ref}

    # ---- learning off: identity mixing equals the fixed V4 exactly
    def new_stage(v):
        return LearnedHigherStage("V4", n_in, sc["sheet"], sc["radius"], bank="mix", mix_radius=v["mix_radius"],
                                  offset=c["offset"], **stage_kw)

    init = new_stage({"mix_radius": 1})
    with torch.no_grad():
        xs = s2_te["original"][:200]
        feat_diff = float((init.features(xs) - fixed.features(xs)).abs().max())
    print(f"learning off: max |feature difference| vs fixed V4 {feat_diff:.2e}", flush=True)
    base = evaluate_seeds(copy.deepcopy(init), "V4 learning off")
    base["max_feature_diff"] = feat_diff
    results["V4_learning_off"] = base

    # ---- trained variants
    histories = {}
    for v in sweep:
        t_len, rname = v.get("seq_len", 2), v.get("range", next(iter(ranges)))
        label = f"{v['mode']}_r{v['mix_radius']}_tether{v['tether']:g}"
        if "range" in v or "seq_len" in v:
            label += f"_{rname}_T{t_len}"
        st = new_stage(v)
        hist = fit_stage_filters(st, sequences(rname, t_len), None,
                                 temporal_weight=v.get("temporal_weight", c["temporal_weights"][v["mode"]]),
                                 tether=v["tether"], white_weight=tr["white_weight"], n_steps=tr["n_steps"],
                                 lr=tr["lr"], batch_size=tr["batch_size"], refit_every=tr["refit_every"],
                                 refit_n=tr["refit_n"], border=sc["border"], seed=seeds[0], tica_kw=tica_kw,
                                 optimizer=tr["optimizer"])
        r = evaluate_seeds(st, f"V4 {label} (drift {hist['drift'][-1]:.3f})")
        dr = hist["drift"]
        tail = max(1, len(dr) // 5)
        r.update(mode=v["mode"], mix_radius=v["mix_radius"], tether=v["tether"], drift=dr[-1], range=rname,
                 seq_len=t_len,
                 drift_last_fifth_share=(dr[-1] - dr[-tail - 1]) / (dr[-1] + 1e-12),
                 change_share=st.bank.offdiagonal_share(),
                 final_losses={k: hist[k][-1] for k in ("bubbles", "white", "tether")})
        print(f"   change share {({k: round(x, 2) for k, x in r['change_share'].items()})}; "
              f"drift added in the last fifth of training {r['drift_last_fifth_share']:.0%}", flush=True)
        results[label], histories[label] = r, hist
        torch.save(st.state_dict(), out / f"v4_{label}.pt")

    # ---- checks (declared before running)
    ch = c["checks"]
    learned = {k: r for k, r in results.items() if "mode" in r}

    def gain(k, key):
        """(gain over learning off, its standard error)."""
        return (learned[k][key] - base[key], math.hypot(learned[k][key + "_se"], base[key + "_se"]))

    def significant(gv, min_gain):
        return gv[0] >= max(min_gain, ch["n_se"] * gv[1])

    keep = [k for k in learned if learned[k]["original"] >= base["original"] - ch["max_accuracy_drop"]]
    bub = [k for k in keep if learned[k]["mode"] == "bubbles"]
    tol = {k: gain(k, "mean_transformed") for k in learned}
    inv = {k: gain(k, "mean_invariance") for k in learned}
    best = max(bub, key=lambda k: tol[k][0]) if bub else None
    checks = {
        "learning_off_equals_fixed_v4": feat_diff <= ch["max_feature_diff"],
        "readout_tolerance_gain": bool(best) and significant(tol[best], ch["min_tolerance_gain"]),
        "invariance_index_gain": any(significant(inv[k], ch["min_invariance_gain"]) for k in bub),
    }
    if best:
        still = best.replace("bubbles", "still", 1)
        if still in learned:
            diff = (learned[best]["mean_transformed"] - learned[still]["mean_transformed"],
                    math.hypot(learned[best]["mean_transformed_se"], learned[still]["mean_transformed_se"]))
            checks["gain_is_temporal"] = significant(diff, ch["min_temporal_margin"])
    r0 = [k for k in keep if learned[k]["mix_radius"] == 0]
    r1 = [k for k in keep if learned[k]["mix_radius"] > 0]
    if r0 and r1:
        checks["mixing_beats_per_channel"] = max(tol[k][0] for k in r1) > max(tol[k][0] for k in r0)
    if "reference_tolerance_gain" in ch:
        wide = [k for k in bub if learned[k]["range"] != ch["reference_range"]]
        if wide:
            checks["matched_range_beats_reference"] = (
                max(tol[k][0] for k in wide) >= ch["reference_tolerance_gain"] + ch["min_range_margin"])
    long_, pairs = [k for k in bub if learned[k]["seq_len"] > 2], [k for k in bub if learned[k]["seq_len"] == 2]
    if long_ and pairs:
        checks["sequences_beat_pairs"] = max(tol[k][0] for k in long_) > max(tol[k][0] for k in pairs)
    summary = {"tolerance_gain": tol, "invariance_gain": inv, "best_bubbles_variant": best}
    if enames:
        summary["extrapolation_gain"] = {k: gain(k, "mean_extrapolation") for k in learned}
    print(json.dumps(summary, indent=2), flush=True)

    # ---- figures
    labels = list(learned)
    if labels:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
        xs_ = range(len(labels))
        cols = [MODE_COLORS[learned[k]["mode"]] for k in labels]
        short = [f"{learned[k]['mode']}\nr={learned[k]['mix_radius']} τ={learned[k]['tether']:g}"
                 + (f"\n{learned[k]['range']} T={learned[k]['seq_len']}" if len(ranges) > 1 or "seq_len" in str(sweep) else "")
                 for k in labels]
        for ax, gains, title, bar in ((axes[0], tol, "Transformed-image accuracy vs learning off", ch["min_tolerance_gain"]),
                                      (axes[1], inv, "Unit invariance index vs learning off", ch["min_invariance_gain"])):
            ax.bar(xs_, [gains[k][0] for k in labels], yerr=[ch["n_se"] * gains[k][1] for k in labels],
                   color=cols, width=0.65, error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})
            ax.axhline(0, color=INK, linewidth=0.8)
            ax.axhline(bar, color=MUTED, linestyle=":", linewidth=1)
            ax.set_xticks(list(xs_), short)
            _style(ax, title + f" (bars: {ch['n_se']:g} SE)", "", "change")
        ax = axes[2]
        for k in labels:
            ax.plot(histories[k]["drift"], color=MODE_COLORS[learned[k]["mode"]],
                    linestyle="-" if learned[k]["mix_radius"] else "--", linewidth=1.3)
            ax.text(len(histories[k]["drift"]) - 1, histories[k]["drift"][-1],
                    f" r={learned[k]['mix_radius']} τ={learned[k]['tether']:g}", fontsize=6, color=MUTED, va="center")
        for mode, col in MODE_COLORS.items():
            ax.plot([], [], color=col, label=mode)
        ax.plot([], [], color=MUTED, linestyle="--", label="own channel (r = 0)")
        ax.legend(fontsize=7, frameon=False, labelcolor=INK)
        _style(ax, "Mixing drift from the identity", "step", "‖A − I‖ / ‖I‖")
        fig.tight_layout()
        fig.savefig(out / f"{Path(c['output_dir']).name}_sweep.png", dpi=120)

    report = {"results": results, "summary": summary, "checks": checks, "passed": all(checks.values()),
              "refit_seeds": seeds}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"checks": checks}, indent=2))
    print(f"{Path(c['output_dir']).name} {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
