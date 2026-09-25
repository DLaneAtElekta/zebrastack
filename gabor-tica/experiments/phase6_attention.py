"""Phase 6 exit check: thalamic gain from generative templates.

1. Top-down path: decoders AIT -> PIT -> V4 fitted on the Phase 5 stack's own
   activity; V4 outputs -> V4 energy channels via the pseudo-inverse of V4's
   whitening + TICA. A category template: clamp AIT to the category's mean
   activity and decode down.
2. Attention field at V4: per-channel gain A = exp(beta z) from the target
   template, applied to V4's Gabor energies before divisive normalization
   (normalization model of attention).
3. Task: a sneaker among 3 other Fashion-MNIST items (28 px) on the 64 px
   canvas vs 4 distractors (absent) vs a sandal/boot lookalike. d' and
   false alarms from a linear readout of AIT (retrained per condition; the
   model's weights are never retrained).

    python experiments/phase6_attention.py [--config configs/phase6.yaml]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gtv.config import load_config  # noqa: E402
from gtv.data import CLASSES, load_fashion_mnist  # noqa: E402
from gtv.generative import decode_down, fit_topdown  # noqa: E402
from gtv.probes.decode import fit_logistic, train_test_split  # noqa: E402
from gtv.probes.sets import clutter_scene  # noqa: E402
from gtv.stages import V2Stage, build_higher_stack, build_stage  # noqa: E402
from gtv.thalamus import feature_gain, feature_similarity_field  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
SEEDS = (0, 1, 2)
KINDS = ("present", "absent", "lookalike")


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def detection(feats: dict, hit_rate: float = 0.8) -> dict:
    """d' (present vs absent) and false alarms (lookalike, absent) at the
    threshold that catches ``hit_rate`` of present scenes; held-out, averaged."""
    out = {"dprime": [], "fa_lookalike": [], "fa_absent": []}
    n = len(feats["present"])
    for seed in SEEDS:
        tr, te = train_test_split(n, 0.6, torch.Generator().manual_seed(seed))
        xtr = torch.cat([feats["present"][tr], feats["absent"][tr]])
        ytr = torch.cat([torch.ones(len(tr)), torch.zeros(len(tr))]).long()
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        with torch.enable_grad():
            model = fit_logistic((xtr - mu) / sd, ytr, 2, l2=1e-2)
        with torch.no_grad():
            sc = {k: (lambda s: s[:, 1] - s[:, 0])(model((feats[k][te] - mu) / sd)) for k in KINDS}
        sp, sa = sc["present"], sc["absent"]
        out["dprime"].append(float((sp.mean() - sa.mean()) / (0.5 * (sp.var() + sa.var())).sqrt()))
        thr = torch.quantile(sp, 1 - hit_rate)
        out["fa_lookalike"].append(float((sc["lookalike"] > thr).float().mean()))
        out["fa_absent"].append(float((sa > thr).float().mean()))
    return {k: sum(v) / len(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase6.yaml"))
    args = ap.parse_args()
    c6 = load_config(args.config)
    c5 = load_config(ROOT / c6["base_config"])
    cfg = load_config(ROOT / c5["base_config"])
    vc, tk = cfg["v2"], c6["task"]
    out = ROOT / c6["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["image_size"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    ckpt_path = ROOT / c6["stack_checkpoint"]
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} not found: run experiments/phase5_hierarchy.py first")

    # ---- the Phase 5 stack, loaded (no refitting)
    v1 = build_stage({"name": "V1", **cfg["v1"]})
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"], sheet=vc["sheet"],
                 radius=vc["radius"], dim=vc["dim"], tica_mode=vc["tica_mode"], **vc["design_B"])
    ck = torch.load(ckpt_path)
    v2.load_state_dict(ck["V2"])
    stack = build_higher_stack(c5["stages"], {"V1": v1.bank.n_filters, "V2": v2.tica.n_units})
    for name, st in stack.items():
        st.load_state_dict(ck[name])
    names = list(stack)  # V4, PIT, AIT

    def low(x):
        m1, s2 = [], []
        with torch.no_grad():
            for xb in x.split(128):
                a = v1(xb)
                m1.append(a)
                s2.append(v2(a))
        return torch.cat(m1), torch.cat(s2)

    def upper(m1, s2, gain=None, spatial=None):
        """``gain``: feature-only (n_second,) gain at the attended stage;
        ``spatial``: (target, beta) for a feature-similarity field computed there."""
        maps = {"V1": m1, "V2": s2}
        prev = s2
        with torch.no_grad():
            for name in names:
                sc = c5["stages"][name]
                out_ = []
                for i, pb in zip(range(0, len(prev), 256), prev.split(256)):
                    skips = [maps[sc["skip"]][i:i + 256]] if sc["skip"] else None
                    g = None
                    if name == tk["attend_stage"]:
                        g = gain
                        if spatial is not None:
                            g = feature_similarity_field(stack[name], pb, templates, spatial[0], spatial[1])
                    out_.append(stack[name](pb, skips, g))
                prev = torch.cat(out_)
                maps[name] = prev
        return maps

    # ---- 1. top-down path and category templates
    x_fit, y_fit = load_fashion_mnist("train", c6["topdown"]["n_fit"], size, seed=5)
    fit_maps = upper(*low(x_fit))
    td = c6["topdown"]
    decoders = fit_topdown({k: fit_maps[k] for k in names}, names, td["kernel"], td["n_iter"], td["lr"],
                           generator=torch.Generator().manual_seed(cfg["seed"]))
    att = stack[tk["attend_stage"]]
    sl = att.second_order_slice
    templates, actual = [], []
    x_val, y_val = load_fashion_mnist("test", 1000, size, seed=6)
    val_maps = upper(*low(x_val))
    for k in range(len(CLASSES)):
        top = fit_maps["AIT"][y_fit == k].mean(0, keepdim=True)
        expected = decode_down(top, decoders, names)[tk["attend_stage"]]
        with torch.no_grad():
            templates.append(att.features_from_outputs(expected)[0, sl].mean((1, 2)))
            m = val_maps[tk["attend_stage"]]
            actual.append(att.features_from_outputs(m[y_val == k])[:, sl].mean((0, 2, 3)))
    templates, actual = torch.stack(templates), torch.stack(actual)
    tc = templates - templates.mean(0)
    ac = actual - actual.mean(0)
    corr = (tc / tc.norm(dim=1, keepdim=True)) @ (ac / ac.norm(dim=1, keepdim=True)).T  # (template k, actual j)
    hits = int((corr.argmax(1) == torch.arange(len(CLASSES))).sum())
    print(f"templates matching their own category: {hits}/10; diag corr median {corr.diag().median():.2f}", flush=True)

    # ---- 2. cluttered scenes
    items, labels = load_fashion_mnist("test", None, raw=True)
    pool = {c: items[labels == c] for c in range(len(CLASSES))}
    others = [c for c in range(len(CLASSES)) if c != tk["target"] and c not in tk["lookalikes"]]

    def pick(c):
        return pool[c][int(torch.randint(len(pool[c]), (1,), generator=gen))]

    def scene(kind):
        dis = [pick(others[int(torch.randint(len(others), (1,), generator=gen))]) for _ in range(tk["n_distractors"])]
        if kind == "present":
            first = pick(tk["target"])
        elif kind == "lookalike":
            first = pick(tk["lookalikes"][int(torch.randint(len(tk["lookalikes"]), (1,), generator=gen))])
        else:
            first = pick(others[int(torch.randint(len(others), (1,), generator=gen))])
        return clutter_scene([first] + dis, size, gen)

    scenes = {k: torch.stack([scene(k) for _ in range(tk["n_per_kind"])]).unsqueeze(1) for k in KINDS}
    lows = {k: low(v) for k, v in scenes.items()}

    def readout(maps, name):
        s = maps[name]
        st = stack[name]
        return F.adaptive_avg_pool2d(torch.cat([s, st.pooled_energy(s)], 1), 2).flatten(1)

    conditions = {f"beta_{b}": (feature_gain(templates, tk["target"], b), None) for b in c6["betas"]}
    conditions["wrong_template"] = (feature_gain(templates, c6["wrong_template"], 1.0), None)
    for b in c6["betas"]:
        if b > 0:
            conditions[f"spatial_beta_{b}"] = (None, (tk["target"], b))
    conditions["spatial_wrong_template"] = (None, (c6["wrong_template"], 1.0))
    results = {}
    for cname, (gain, spatial) in conditions.items():
        maps = {k: upper(*lows[k], gain, spatial) for k in KINDS}
        results[cname] = {stage: detection({k: readout(maps[k], stage) for k in KINDS})
                          for stage in ("V4", "AIT")}
        if gain is not None:
            results[cname]["gain_range"] = [float(gain.min()), float(gain.max())]
        r = results[cname]["AIT"]
        print(f"{cname}: AIT d' {r['dprime']:.2f} FA(lookalike) {r['fa_lookalike']:.2f} | "
              f"V4 d' {results[cname]['V4']['dprime']:.2f}", flush=True)

    # ---- tuning shift (Cukur et al.): V4 units' category tuning with vs without attention (beta 1)
    base_maps = upper(*low(x_val))
    att_maps = upper(*low(x_val), conditions["beta_1.0"][0])

    def tuning(maps):
        e = maps[tk["attend_stage"]].pow(2).mean((2, 3))
        return torch.stack([e[y_val == c].mean(0) for c in range(len(CLASSES))])  # (K, units)

    t0, t1 = tuning(base_maps), tuning(att_maps)
    rel0 = t0[tk["target"]] / t0.mean(0)
    rel1 = t1[tk["target"]] / t1.mean(0)
    shift = {"fraction_units_shifting_toward_target": float((rel1 > rel0).float().mean()),
             "median_log_ratio_change": float((rel1 / rel0).log().median())}
    print(f"tuning shift: {shift}", flush=True)

    ch = c6["checks"]
    base = results["beta_0.0"]["AIT"]["dprime"]
    best_beta = max((k for k in results if k.startswith("beta_") and k != "beta_0.0"),
                    key=lambda k: results[k]["AIT"]["dprime"])
    best_spatial = max((k for k in results if k.startswith("spatial_beta_")),
                       key=lambda k: results[k]["AIT"]["dprime"])
    sp_gain = results[best_spatial]["AIT"]["dprime"] - base
    sp_wrong = results["spatial_wrong_template"]["AIT"]["dprime"] - base
    checks = {
        "templates_category_specific": hits >= ch["min_template_hits"],
        "attention_raises_dprime": results[best_beta]["AIT"]["dprime"] - base >= ch["min_dprime_gain"],
        "spatial_attention_raises_dprime": sp_gain >= ch["spatial_min_dprime_gain"],
        "spatial_attention_is_target_specific": sp_gain - sp_wrong >= ch["spatial_min_specificity"],
    }

    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
    ax = axes[0]
    im = ax.imshow(corr.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(10), [c[:6] for c in CLASSES], rotation=60, fontsize=6)
    ax.set_yticks(range(10), [c[:6] for c in CLASSES], fontsize=6)
    ax.set_xlabel("actual V4 activity (held out)", fontsize=8, color=MUTED)
    ax.set_ylabel("decoded template (AIT clamped)", fontsize=8, color=MUTED)
    ax.set_title(f"Top-down templates vs actual ({hits}/10 on the diagonal)", fontsize=9, color=INK, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04)
    cn = list(results)
    for ax, key, title in ((axes[1], "dprime", "Sneaker-in-clutter d′ (AIT readout)"),
                           (axes[2], "fa_lookalike", "False alarms on sandal/boot scenes")):
        vals = [results[c]["AIT"][key] for c in cn]
        cols = ["#6b6a63" if c == "beta_0.0" else "#eb6834" if "wrong" in c
                else "#1baf7a" if c.startswith("spatial") else "#2a78d6" for c in cn]
        ax.bar(range(len(cn)), vals, color=cols, width=0.65)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(cn)), [c.replace("spatial_", "spatial\n").replace("beta_", "β=")
                                       .replace("wrong_template", "bag, β=1") for c in cn], fontsize=6)
        _style(ax, title)
    fig.tight_layout()
    fig.savefig(out / "phase6_attention.png", dpi=120)
    fig2, ax = plt.subplots(figsize=(9, 2.4))
    ax.imshow(torch.stack([scenes[k][i, 0] for k in KINDS for i in range(4)]).view(3, 4, size, size)
              .permute(0, 2, 1, 3).reshape(3 * size, 4 * size).numpy(), cmap="gray")
    ax.set_yticks([size * (i + 0.5) for i in range(3)], ["sneaker present", "absent", "lookalike"], fontsize=7)
    ax.set_xticks([])
    fig2.tight_layout()
    fig2.savefig(out / "phase6_scenes.png", dpi=120)

    report = {"template_hits": hits, "template_corr": corr.tolist(), "results": results, "tuning_shift": shift,
              "best_beta": best_beta, "best_spatial": best_spatial, "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"checks": checks, "best_beta": best_beta}, indent=2))
    print(f"Phase 6 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
