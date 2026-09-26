"""Phase 6 exit check: thalamic gain from generative templates.

(``template_source: bottomup`` in the config replaces the generated templates
with the attended stage's own category-mean features: a control that
separates template quality from the stack.)

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
from gtv.probes.decode import fit_logistic, linear_decode, train_test_split  # noqa: E402
from gtv.probes.sets import clutter_scene  # noqa: E402
from gtv.stages import V2Stage, build_higher_stack, build_stage  # noqa: E402
from gtv.thalamus import expectation, feature_gain, feature_similarity_field, pass_through_gain  # noqa: E402

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


def detection(feats: dict, hit_rate: float = 0.8, train_feats: dict | None = None) -> dict:
    """d' (present vs absent) and false alarms (lookalike, absent) at the
    threshold that catches ``hit_rate`` of present scenes; held-out, averaged.
    ``train_feats``: fit the readout (and its standardization) on these instead
    (a fixed readout, e.g. trained without attention) and test on ``feats``."""
    out = {"dprime": [], "fa_lookalike": [], "fa_absent": []}
    n = len(feats["present"])
    src = train_feats if train_feats is not None else feats
    for seed in SEEDS:
        tr, te = train_test_split(n, 0.6, torch.Generator().manual_seed(seed))
        xtr = torch.cat([src["present"][tr], src["absent"][tr]])
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
    ap.add_argument("--calibrate-noise", nargs="*", type=float, metavar="T",
                    help="only report no-attention AIT d' for each response-noise scale T")
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

    noise_T = c6.get("response_noise_T")

    def upper(m1, s2, gain=None, spatial=None, noise=None, noise_seed=0, pass_gain=None, expect=None):
        """``gain``: feature-only (n_second,) gain at the attended stage;
        ``spatial``: (target, beta) for a feature-similarity field computed there;
        ``noise``: response-noise scale T at the attended stage (same draws for a
        given ``noise_seed``, so conditions are compared on identical noise);
        ``pass_gain``: (n_first,) gain on the attended stage's pass-through channels;
        ``expect``: expectation channel (predictive subtraction) at the attended stage."""
        maps = {"V1": m1, "V2": s2}
        prev = s2
        ngen = torch.Generator().manual_seed(noise_seed)
        with torch.no_grad():
            for name in names:
                sc = c5["stages"][name]
                out_ = []
                for i, pb in zip(range(0, len(prev), 256), prev.split(256)):
                    skips = [maps[sc["skip"]][i:i + 256]] if sc["skip"] else None
                    g, nt, pg, ex = None, None, None, None
                    if name == tk["attend_stage"]:
                        g, nt, pg, ex = gain, noise, pass_gain, expect
                        if spatial is not None:
                            g = feature_similarity_field(stack[name], pb, templates, spatial[0], spatial[1])
                    out_.append(stack[name](pb, skips, g, nt, ngen, pg, ex))
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
    source = c6.get("template_source", "topdown")
    if source == "bottomup":
        # control: category means of the attended stage's own second-order
        # features (log-normalized pooled energies) for the fitting images,
        # i.e. an ideal feature template that bypasses the top-down path
        idx = names.index(tk["attend_stage"])
        below = fit_maps["V2" if idx == 0 else names[idx - 1]]
        with torch.no_grad():
            f = torch.cat([att.features(b)[:, sl].mean((2, 3)) for b in below.split(256)])
        topdown_templates = templates
        templates = torch.stack([f[y_fit == k].mean(0) for k in range(len(CLASSES))])
    elif source != "topdown":
        raise ValueError(f"unknown template_source {source!r}")
    tc = templates - templates.mean(0)
    ac = actual - actual.mean(0)
    corr = (tc / tc.norm(dim=1, keepdim=True)) @ (ac / ac.norm(dim=1, keepdim=True)).T  # (template k, actual j)
    hits = int((corr.argmax(1) == torch.arange(len(CLASSES))).sum())
    print(f"{source} templates matching their own category: {hits}/10; diag corr median {corr.diag().median():.2f}",
          flush=True)
    if source == "bottomup":
        zt = lambda t: (t - t.mean(0)) / (t.std(0) + 1e-6)  # noqa: E731
        agree = float(torch.nn.functional.cosine_similarity(zt(templates)[tk["target"]],
                                                            zt(topdown_templates)[tk["target"]], dim=0))
        print(f"target gain pattern, bottom-up vs top-down (cosine of z): {agree:.2f}", flush=True)

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

    if args.calibrate_noise is not None:
        for T in args.calibrate_noise or [1.0, 3.0, 10.0, 30.0]:
            maps = {k: upper(*lows[k], noise=T, noise_seed=i) for i, k in enumerate(KINDS)}
            r = detection({k: readout(maps[k], "AIT") for k in KINDS})
            print(f"noise T {T}: no-attention AIT d' {r['dprime']:.2f}", flush=True)
        return

    conditions = {f"beta_{b}": (feature_gain(templates, tk["target"], b), None, None, None) for b in c6["betas"]}
    conditions["wrong_template"] = (feature_gain(templates, c6["wrong_template"], 1.0), None, None, None)
    if c6.get("spatial", True):
        for b in c6["betas"]:
            if b > 0:
                conditions[f"spatial_beta_{b}"] = (None, (tk["target"], b), None, None)
        conditions["spatial_wrong_template"] = (None, (c6["wrong_template"], 1.0), None, None)
    if c6.get("pass_betas") or c6.get("both_betas"):
        # gain on the pass-through channels too: bottom-up templates of their energy
        # (category means over the fitting images), with a fixed power budget
        idx = names.index(tk["attend_stage"])
        below = fit_maps["V2" if idx == 0 else names[idx - 1]]
        with torch.no_grad():
            e1 = torch.cat([att.features(b)[:, :att.n_first].pow(2).mean((2, 3)) for b in below.split(256)])
        pass_templates = torch.stack([e1[y_fit == k].mean(0) for k in range(len(CLASSES))])
        wb = c6.get("wrong_beta", 1.0)
        for b in c6.get("pass_betas", []):
            conditions[f"pass_beta_{b}"] = (None, None, pass_through_gain(pass_templates, tk["target"], b), None)
        for b in c6.get("both_betas", []):
            conditions[f"both_beta_{b}"] = (feature_gain(templates, tk["target"], b), None,
                                           pass_through_gain(pass_templates, tk["target"], b), None)
        conditions["both_wrong_template"] = (feature_gain(templates, c6["wrong_template"], wb), None,
                                             pass_through_gain(pass_templates, c6["wrong_template"], wb), None)
    ex = c6.get("expectation")
    if ex:
        # Phase 7: expectation (predictive subtraction), alone and with attention at beta ex["with_beta"]
        att_gain = feature_gain(templates, tk["target"], ex["with_beta"])
        for a in ex["alphas"]:
            conditions[f"expect_a{a}"] = (None, None, None, expectation(templates, tk["target"], a))
            conditions[f"att_expect_a{a}"] = (att_gain, None, None, expectation(templates, tk["target"], a))
        a0 = ex["control_alpha"]
        conditions["att_expect_uniform"] = (att_gain, None, None, expectation(templates, tk["target"], a0, spatial=False))
        conditions["att_expect_wrong"] = (att_gain, None, None, expectation(templates, c6["wrong_template"], a0))
    results = {}
    reference = None  # AIT features without attention: the fixed readout's training data
    if noise_T is not None:
        results["noiseless_no_attention"] = {"AIT": detection(
            {k: readout(upper(*lows[k]), "AIT") for k in KINDS})}
    for cname, (gain, spatial, pgain, expct) in conditions.items():
        maps = {k: upper(*lows[k], gain, spatial, noise_T, i, pgain, expct) for i, k in enumerate(KINDS)}
        feats_ait = {k: readout(maps[k], "AIT") for k in KINDS}
        if cname == "beta_0.0":
            reference = feats_ait
        results[cname] = {stage: detection({k: readout(maps[k], stage) for k in KINDS})
                          for stage in ("V4", "AIT")}
        results[cname]["AIT_fixed_readout"] = detection(feats_ait, train_feats=reference)
        if gain is not None:
            results[cname]["gain_range"] = [float(gain.min()), float(gain.max())]
        if pgain is not None:
            results[cname]["pass_gain_range"] = [float(pgain.min()), float(pgain.max())]
        r = results[cname]["AIT"]
        print(f"{cname}: AIT d' {r['dprime']:.2f} FA(lookalike) {r['fa_lookalike']:.2f} | fixed-readout d' "
              f"{results[cname]['AIT_fixed_readout']['dprime']:.2f} | V4 d' {results[cname]['V4']['dprime']:.2f}",
              flush=True)

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

    kok = None
    if ex:
        # expectation suppression (Kok et al.): the attended stage's second-order
        # responses to clean images of each category, with vs without the target
        # expectation (response = distance from the no-expectation mean pattern),
        # and target-vs-lookalike decoding from those responses
        fn = expectation(templates, tk["target"], ex["control_alpha"])
        idx = names.index(tk["attend_stage"])
        below_val = val_maps["V2" if idx == 0 else names[idx - 1]]
        with torch.no_grad():
            f0 = torch.cat([att.features(b)[:, sl] for b in below_val.split(256)])
            f1 = torch.cat([att.features(b, expect=fn)[:, sl] for b in below_val.split(256)])
        mu = f0.mean(0, keepdim=True)
        r0, r1 = (f0 - mu).flatten(1).norm(dim=1), (f1 - mu).flatten(1).norm(dim=1)
        ratio = {CLASSES[c]: float(r1[y_val == c].mean() / r0[y_val == c].mean()) for c in range(len(CLASSES))}
        others = [ratio[CLASSES[c]] for c in range(len(CLASSES)) if c != tk["target"]]
        mask = (y_val == tk["target"]) | torch.isin(y_val, torch.tensor(tk["lookalikes"]))
        yl = (y_val[mask] == tk["target"]).long()
        dec = [sum(linear_decode(F.adaptive_avg_pool2d(f[mask], 2).flatten(1), yl, seed=sd) for sd in SEEDS) / len(SEEDS)
               for f in (f0, f1)]
        kok = {"response_ratio": ratio, "target_ratio": ratio[CLASSES[tk["target"]]],
               "other_ratio_mean": sum(others) / len(others),
               "target_vs_lookalike_decoding": {"without": dec[0], "with": dec[1]}}
        print(f"expectation suppression: target response x{kok['target_ratio']:.3f}, others x{kok['other_ratio_mean']:.3f}; "
              f"target vs lookalike decoding {dec[0]:.3f} -> {dec[1]:.3f}", flush=True)

    ch = c6["checks"]
    base = results["beta_0.0"]["AIT"]["dprime"]
    cond_names = [k for k in results if k != "noiseless_no_attention"]
    best_beta = max((k for k in results if k.startswith("beta_") and k != "beta_0.0"),
                    key=lambda k: results[k]["AIT"]["dprime"])
    checks = {
        "templates_category_specific": hits >= ch["min_template_hits"],
        "attention_raises_dprime": results[best_beta]["AIT"]["dprime"] - base >= ch["min_dprime_gain"],
    }
    spatial_keys = [k for k in results if k.startswith("spatial_beta_")]
    if spatial_keys:
        best_spatial = max(spatial_keys, key=lambda k: results[k]["AIT"]["dprime"])
        sp_gain = results[best_spatial]["AIT"]["dprime"] - base
        sp_wrong = results["spatial_wrong_template"]["AIT"]["dprime"] - base
        checks["spatial_attention_raises_dprime"] = sp_gain >= ch["spatial_min_dprime_gain"]
        checks["spatial_attention_is_target_specific"] = sp_gain - sp_wrong >= ch["spatial_min_specificity"]
    combined = [k for k in results if k.startswith(("pass_beta_", "both_beta_"))]
    if combined:
        best_comb = max(combined, key=lambda k: results[k]["AIT"]["dprime"])
        comb_gain = results[best_comb]["AIT"]["dprime"] - base
        checks["combined_attention_raises_dprime"] = comb_gain >= ch["min_dprime_gain"]
        both_wrong = results["both_wrong_template"]["AIT"]["dprime"] - base
        best_both = max((k for k in combined if k.startswith("both_")), key=lambda k: results[k]["AIT"]["dprime"],
                        default=None)
        if best_both:
            checks["combined_is_target_specific"] = (
                results[best_both]["AIT"]["dprime"] - base - both_wrong >= ch["feature_min_specificity"])
    if ex:
        att_base = results[f"beta_{ex['with_beta']}"]["AIT"]
        cands = [k for k in results if k.startswith("att_expect_a")]
        kept = [k for k in cands if results[k]["AIT"]["dprime"] >= att_base["dprime"] - ch["max_dprime_drop"]]
        best_ex = min(kept, key=lambda k: results[k]["AIT"]["fa_lookalike"]) if kept else None
        checks["expectation_lowers_false_alarms"] = bool(best_ex) and (
            results[best_ex]["AIT"]["fa_lookalike"] <= att_base["fa_lookalike"] - ch["min_fa_drop"])
        checks["expectation_suppression"] = kok["target_ratio"] < kok["other_ratio_mean"]
        checks["expectation_sharpens"] = (kok["target_vs_lookalike_decoding"]["with"]
                                          >= kok["target_vs_lookalike_decoding"]["without"])
    if "feature_min_specificity" in ch:
        fe_wrong = results["wrong_template"]["AIT"]["dprime"] - base
        checks["attention_is_target_specific"] = (results[best_beta]["AIT"]["dprime"] - base) - fe_wrong >= ch["feature_min_specificity"]
    if noise_T is not None and "max_noisy_baseline_fraction" in ch:
        checks["noise_bottleneck_costs_information"] = (
            base <= ch["max_noisy_baseline_fraction"] * results["noiseless_no_attention"]["AIT"]["dprime"])

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
    cn = cond_names
    for ax, key, title in ((axes[1], "dprime", "Sneaker-in-clutter d′ (AIT readout, retrained)"),
                           (axes[2], "dprime", "d′ with a fixed readout (trained without attention)")):
        src = "AIT" if "retrained" in title else "AIT_fixed_readout"
        vals = [results[c][src][key] for c in cn]
        if "noiseless_no_attention" in results:
            ax.axhline(results["noiseless_no_attention"]["AIT"]["dprime"], color=MUTED, linestyle="--", linewidth=1)
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

    report = {"expectation_suppression": kok, "template_source": source, "template_hits": hits, "template_corr": corr.tolist(), "results": results, "tuning_shift": shift,
              "best_beta": best_beta, "best_spatial": best_spatial if spatial_keys else None, "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"checks": checks, "best_beta": best_beta}, indent=2))
    print(f"Phase 6 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
