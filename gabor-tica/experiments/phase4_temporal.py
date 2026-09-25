"""Phase 4 exit check: temporal coherence on simulated drift sequences.

Design B V2 stages fitted on consecutive-frame pairs with still TICA (frames
independent), bubbles TICA (energy pooled over the sheet neighborhood and both
frames), or TICA + temporal coherence of activity levels (Hurri & Hyvarinen),
all from the same random start.

Invariance: correlation, across held-out images, of each unit's pooled energy
for an image and its shifted / rotated / rescaled version (same location).
Selectivity: Phase 2 probe decoding (signed s + pooled energy readout) and
orientation selectivity of pooled energy on oriented textures.
Exit: some temporal model improves invariance while keeping selectivity.

    python experiments/phase4_temporal.py [--config configs/phase4.yaml]
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
from gtv.data import REPO_PHOTOS, affine, center_crop, drift_pairs, load_grayscale, random_patches  # noqa: E402
from gtv.probes import linear_decode  # noqa: E402
from gtv.probes.sets import junction_set, second_order_set, texture_set  # noqa: E402
from gtv.stages import V2Stage, build_stage  # noqa: E402
from gtv.stages.tica import torus_distance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
COLORS = {"still": "#6b6a63", "bubbles_w1": "#86b6ef", "bubbles_w3": "#2a78d6",
          "coherence_0.3": "#f2b48f", "coherence_1": "#eb6834", "coherence_3": "#b54a1f"}
SEEDS = (0, 1, 2)


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


def window_mean(y, w=4):
    lo = (y.shape[2] - w) // 2
    return y[:, :, lo : lo + w, lo : lo + w].mean((2, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase4.yaml"))
    args = ap.parse_args()
    c4 = load_config(args.config)
    cfg = load_config(ROOT / c4["base_config"])
    vc, d = cfg["v2"], c4["data"]
    out = ROOT / c4["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["image_size"]
    gen = torch.Generator().manual_seed(cfg["seed"])

    v1 = build_stage({"name": "V1", **cfg["v1"]})
    photos = load_grayscale(REPO_PHOTOS)
    train_photos, test_photos = photos[: d["train_photos"]], photos[d["train_photos"] :]
    with torch.no_grad():
        pairs = drift_pairs(train_photos, d["n_pairs"], size, generator=gen, **d["drift"])
        m0, m1 = v1(pairs[:, 0]), v1(pairs[:, 1])
        test_pairs = drift_pairs(test_photos, d["n_test_pairs"], size, generator=gen, **d["drift"])
        t0, t1 = v1(test_pairs[:, 0]), v1(test_pairs[:, 1])

    # ---- fit the models
    stages, hists = {}, {}
    for name, (mode, weight) in c4["models"].items():
        st = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"],
                     sheet=vc["sheet"], radius=vc["radius"], dim=vc["dim"], **vc["design_B"])
        hists[name] = st.fit_temporal(m0, m1, mode, weight, seed=cfg["seed"], **c4["fit"])
        stages[name] = st
        print(f"fitted {name}", flush=True)
    front = stages["still"].front  # identical fixed front end in every model

    def feats_of(v1maps):
        with torch.no_grad():
            return torch.cat([front(x) for x in v1maps.split(64)])

    def outputs(st, f):
        m, b = st.tica.affine
        s = torch.einsum("nd,bdhw->bnhw", m, f) + b.view(1, -1, 1, 1)
        return s, st.pooled_energy(s)

    def unit_corr(a, b):
        a = a[..., 2:-2, 2:-2].transpose(0, 1).flatten(1)
        b = b[..., 2:-2, 2:-2].transpose(0, 1).flatten(1)
        a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-8)

    # ---- held-out drift coherence and invariance tests
    f_t0, f_t1 = feats_of(t0), feats_of(t1)
    margin = 16
    with torch.no_grad():
        big = random_patches(test_photos, d["n_invariance"], size + 2 * margin, gen)
        base_img = center_crop(big, size)
        f_base = feats_of(v1(base_img))
        f_tr = {}
        n = len(big)
        for tname, (sx, rot, sc) in c4["invariance_tests"].items():
            moved = affine(big, torch.tensor([[sx, 0.0]]).expand(n, 2), torch.full((n,), rot), torch.full((n,), sc))
            f_tr[tname] = feats_of(v1(center_crop(moved, size)))

    # ---- selectivity probes (shared front-end features)
    p = cfg["probes"]
    x1, y1 = texture_set(p, size, gen)
    probe_sets = {"texture_first_order": (x1, y1),
                  "second_order_0.0156": second_order_set({**p, "cm_envelope_freq": 0.0156}, size, gen),
                  "second_order_0.0625": second_order_set({**p, "cm_envelope_freq": 0.0625}, size, gen)}
    xj, yj, _ = junction_set(p, size, gen)
    probe_sets["junctions"] = (xj, yj)
    with torch.no_grad():
        probe_feats = {k: feats_of(v1(x)) for k, (x, _) in probe_sets.items()}

    rows = vc["sheet"]
    dsheet = torus_distance(rows, rows)
    off = ~torch.eye(rows * rows, dtype=torch.bool)
    results = {}
    for name, st in stages.items():
        with torch.no_grad():
            s0, e0 = outputs(st, f_t0)
            s1, e1 = outputs(st, f_t1)
            r = {"drift_coherence_energy": float(unit_corr(e0, e1).median()),
                 "drift_coherence_raw": float(unit_corr(s0, s1).median())}
            _, eb = outputs(st, f_base)
            inv = {}
            for tname, f in f_tr.items():
                _, et = outputs(st, f)
                inv[tname] = float(unit_corr(eb, et).median())
            r["invariance"] = inv
            r["invariance_mean"] = sum(inv.values()) / len(inv)
            # orientation selectivity of pooled energy on the 8 oriented textures
            _, ep = outputs(st, probe_feats["texture_first_order"])
            resp = window_mean(ep)
            means = torch.stack([resp[y1 == c].mean(0) for c in range(8)])
            ang = torch.arange(8) * 2 * math.pi / 8
            vec = (means * torch.exp(1j * ang)[:, None]).sum(0) / means.sum(0)
            r["orientation_selectivity_median"] = float(vec.abs().median())
            # topography (raw s energies on held-out drift frames)
            ss = s0[..., 2:-2, 2:-2].permute(0, 2, 3, 1).reshape(-1, rows * rows)
            ec = torch.corrcoef(ss.pow(2).T)
            r["topography_near_far"] = float(ec[(dsheet == 1) & off].mean() / ec[(dsheet == rows // 2) & off].mean())
        dec = {}
        for k, (_, y) in probe_sets.items():
            with torch.no_grad():
                s, e = outputs(st, probe_feats[k])
            feat = window_mean(torch.cat([s, e], 1))
            dec[k] = sum(linear_decode(feat, y, seed=sd) for sd in SEEDS) / len(SEEDS)
        r["decoding"] = dec
        results[name] = r
        print(f"{name}: drift coherence {r['drift_coherence_energy']:.3f} | invariance {r['invariance_mean']:.3f} "
              f"{ {k: round(v, 3) for k, v in inv.items()} } | orient sel {r['orientation_selectivity_median']:.3f} | "
              f"decoding { {k: round(v, 2) for k, v in dec.items()} }", flush=True)

    # ---- checks (declared in the config before running)
    ch = c4["checks"]
    ref = results["still"]
    per_model = {}
    for name, r in results.items():
        if name == "still":
            continue
        inv_ok = r["invariance_mean"] - ref["invariance_mean"] >= ch["min_invariance_gain"]
        dec_ok = all(r["decoding"][k] >= ref["decoding"][k] - ch["max_decoding_drop"] for k in r["decoding"])
        sel_ok = r["orientation_selectivity_median"] >= ch["min_selectivity_ratio"] * ref["orientation_selectivity_median"]
        per_model[name] = {"invariance_improved": inv_ok, "decoding_kept": dec_ok, "selectivity_kept": sel_ok,
                           "both": inv_ok and dec_ok and sel_ok}
    checks = {"some_temporal_model_more_invariant_and_still_selective": any(v["both"] for v in per_model.values())}

    # ---- figures
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    ax = axes[0]
    tests = list(c4["invariance_tests"])
    width = 0.8 / len(names)
    for i, nme in enumerate(names):
        xs = [j + (i - (len(names) - 1) / 2) * width for j in range(len(tests))]
        ax.bar(xs, [results[nme]["invariance"][t] for t in tests], width * 0.9, color=COLORS[nme], label=nme)
    ax.set_xticks(range(len(tests)), [t.replace("_", " ") for t in tests], fontsize=7)
    ax.set_ylim(0, 1)
    _style(ax, "Invariance (median unit corr, image vs transformed)")
    ax.legend(fontsize=6, frameon=False, ncol=2, labelcolor=INK)
    ax = axes[1]
    probes = list(results["still"]["decoding"])
    for i, nme in enumerate(names):
        xs = [j + (i - (len(names) - 1) / 2) * width for j in range(len(probes))]
        ax.bar(xs, [results[nme]["decoding"][k] for k in probes], width * 0.9, color=COLORS[nme])
    ax.set_xticks(range(len(probes)), [k.replace("_", "\n") for k in probes], fontsize=6)
    ax.set_ylim(0, 1.05)
    _style(ax, "Selectivity: probe decoding (held out)")
    ax = axes[2]
    for nme in names:
        ax.scatter(results[nme]["invariance_mean"], results[nme]["orientation_selectivity_median"], s=60,
                   color=COLORS[nme], zorder=3)
        ax.annotate(nme, (results[nme]["invariance_mean"], results[nme]["orientation_selectivity_median"]),
                    fontsize=7, color=INK, xytext=(4, 4), textcoords="offset points")
    ax.grid(True, color=GRID, linewidth=0.6)
    _style(ax, "Invariance vs orientation selectivity", "mean invariance", "median orientation selectivity")
    fig.tight_layout()
    fig.savefig(out / "phase4_invariance_selectivity.png", dpi=120)

    report = {"results": results, "per_model_checks": per_model, "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({k: s.state_dict() for k, s in stages.items()}, out / "phase4_stages.pt")
    print(json.dumps({"per_model_checks": per_model, "checks": checks}, indent=2))
    print(f"Phase 4 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
