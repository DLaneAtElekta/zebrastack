"""Phase 5 exit check: V4 -> PIT -> AIT recognition stack.

V1 and the Design B V2 are as validated on natural photos. V4, PIT and AIT
repeat the Design B block (fixed Gabors on every input channel, energy,
normalization, log, pooling, group-whitened TICA), with skips V1 -> V4 and
V2 -> PIT and larger TICA neighborhoods higher up. They are fitted greedily,
without labels, on Fashion-MNIST images.

Reports, per stage: linear category decoding (2x2-pooled [s, pooled energy]),
tolerance to shift / rotation / scale (decoder trained on originals), V4 vs
V2 curvature coding, and AIT sheet category clustering (preference from each
unit's own unpooled energy; neighbor agreement vs shuffled sheets).
Exit: AIT decoding well above chance and emergent topographic category clusters.

    python experiments/phase5_hierarchy.py [--config configs/phase5.yaml]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gtv.config import load_config  # noqa: E402
from gtv.data import CLASSES, REPO_PHOTOS, SUPERORDINATE, affine, load_fashion_mnist, load_grayscale, random_patches  # noqa: E402
from gtv.probes import curvature_fragment  # noqa: E402
from gtv.probes.decode import _standardize, fit_logistic, linear_decode  # noqa: E402
from gtv.stages import HigherStage, V2Stage, build_stage  # noqa: E402
from gtv.stages.tica import torus_distance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
GROUP_COLORS = {"tops": "#2a78d6", "footwear": "#eb6834", "other": "#1baf7a"}
ABBR = ["Ts", "Tr", "Pu", "Dr", "Co", "Sa", "Sh", "Sn", "Ba", "Bo"]
STAGE_ORDER = ["pixels", "V1", "V2", "V4", "PIT", "AIT"]
SEEDS = (0, 1, 2)


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
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase5.yaml"))
    args = ap.parse_args()
    c5 = load_config(args.config)
    cfg = load_config(ROOT / c5["base_config"])
    vc, d = cfg["v2"], c5["data"]
    out = ROOT / c5["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["image_size"]
    gen = torch.Generator().manual_seed(cfg["seed"])

    # ---- V1 and the Phase 2 Design B V2 (natural photos)
    v1 = build_stage({"name": "V1", **cfg["v1"]})
    with torch.no_grad():
        nat = v1(random_patches(load_grayscale(REPO_PHOTOS), cfg["train"]["n_patches"], size, gen))
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"], sheet=vc["sheet"],
                 radius=vc["radius"], dim=vc["dim"], tica_mode=vc["tica_mode"], **vc["design_B"])
    v2.fit(nat, n_iter=vc["n_iter"], seed=cfg["seed"])

    # ---- category data
    x_fit, _ = load_fashion_mnist("train", d["n_fit"], size, seed=1)
    x_tr, y_tr = load_fashion_mnist("train", d["n_decode_train"], size, seed=2)
    x_te, y_te = load_fashion_mnist("test", d["n_decode_test"], size, seed=3)

    def low(x):
        """V1 maps and V2 TICA outputs, in chunks."""
        m1, s2 = [], []
        with torch.no_grad():
            for xb in x.split(128):
                a = v1(xb)
                m1.append(a)
                s2.append(v2(a))
        return torch.cat(m1), torch.cat(s2)

    stages, whitening_floored = {}, {}
    m1_fit, s2_fit = low(x_fit)
    below = {"V1": m1_fit, "V2": s2_fit}
    prev_name, prev = "V2", s2_fit
    for name, sc in c5["stages"].items():
        skip_src = sc["skip"]
        st = HigherStage(name, prev.shape[1], sc["sheet"], sc["radius"],
                         skip_channels=below[skip_src].shape[1] if skip_src else 0)
        skips = [below[skip_src]] if skip_src else None
        st.fit(prev, skips, border=sc["border"], seed=cfg["seed"], **c5["tica"])
        with torch.no_grad():
            nxt = torch.cat([st(pb, [s[i:i + 256] for s in skips] if skips else None)
                             for i, pb in zip(range(0, len(prev), 256), prev.split(256))])
        stages[name] = st
        below[name] = nxt
        wh = st.tica.whitener
        floored = {"first_order": wh.parts[0].n_floored, "second_order": wh.parts[1].n_floored,
                   "joint": wh.joint.n_floored}
        prev_name, prev = name, nxt
        whitening_floored[name] = floored
        print(f"fitted {name}: {tuple(nxt.shape)}; whitening components floored {floored}", flush=True)

    def run(x):
        """Per-stage maps for images x: V1 log energy; V2..AIT TICA outputs s."""
        m1, s2 = low(x)
        maps = {"V1": m1, "V2": s2}
        prev = s2
        for name, sc in c5["stages"].items():
            skips = [maps[sc["skip"]]] if sc["skip"] else None
            with torch.no_grad():
                prev = torch.cat([stages[name](pb, [s[i:i + 256] for s in skips] if skips else None)
                                  for i, pb in zip(range(0, len(prev), 256), prev.split(256))])
            maps[name] = prev
        return maps

    def energy_of(name, s):
        if name == "V1":
            return s
        st = v2 if name == "V2" else stages[name]
        return st.pooled_energy(s)

    def readout(name, maps, x):
        """2x2-pooled [s, pooled energy] (V1: log energy), flattened; pixels: the 28x28 content."""
        if name == "pixels":
            return F.adaptive_avg_pool2d(x, 28).flatten(1)
        s = maps[name]
        feat = s if name == "V1" else torch.cat([s, energy_of(name, s)], 1)
        return F.adaptive_avg_pool2d(feat, 2).flatten(1)

    maps_tr, maps_te = run(x_tr), run(x_te)
    feats_tr = {k: readout(k, maps_tr, x_tr) for k in STAGE_ORDER}
    feats_te = {k: readout(k, maps_te, x_te) for k in STAGE_ORDER}

    def train_eval(k, test_sets):
        xtr = feats_tr[k]
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        with torch.enable_grad():
            model = fit_logistic((xtr - mu) / sd, y_tr, len(CLASSES), l2=1e-2)
        res = {}
        with torch.no_grad():
            for tname, (xt, yt) in test_sets.items():
                res[tname] = float((model((xt - mu) / sd).argmax(1) == yt).float().mean())
        return res

    # tolerance: same decoder, transformed test images
    n = len(x_te)
    transformed = {}
    for tname, (sx, rot, scl) in c5["transforms"].items():
        with torch.no_grad():
            xt = affine(x_te, torch.tensor([[sx, 0.0]]).expand(n, 2), torch.full((n,), rot), torch.full((n,), scl))
        mt = run(xt)
        transformed[tname] = {k: readout(k, mt, xt) for k in STAGE_ORDER}
    decoding = {}
    for k in STAGE_ORDER:
        tests = {"original": (feats_te[k], y_te)}
        tests.update({tn: (tf[k], y_te) for tn, tf in transformed.items()})
        r = train_eval(k, tests)
        r["tolerance"] = {tn: r[tn] / r["original"] for tn in transformed}
        decoding[k] = r
        print(f"{k}: accuracy {r['original']:.3f}  tolerance { {t: round(v, 2) for t, v in r['tolerance'].items()} }",
              flush=True)

    # ---- V4 vs V2 curvature coding
    cv = c5["curvature"]
    imgs, labels = [], []
    for ci, k in enumerate(cv["levels"]):
        for _ in range(cv["per_class"]):
            th = float(torch.rand(1, generator=gen)) * 2 * math.pi
            sign = 1 if float(torch.rand(1, generator=gen)) > 0.5 else -1
            img = curvature_fragment(size, sign * k, th)
            dx, dy = (torch.randint(-3, 4, (2,), generator=gen)).tolist()
            imgs.append(torch.roll(img, (dy, dx), (0, 1)))
            labels.append(ci)
    xc = torch.stack(imgs).unsqueeze(1)
    xc = xc - xc.mean((2, 3), keepdim=True)
    yc = torch.tensor(labels)
    mc = run(xc)
    curvature = {k: sum(linear_decode(readout(k, mc, xc), yc, seed=s) for s in SEEDS) / len(SEEDS)
                 for k in ("V1", "V2", "V4", "PIT")}
    print(f"curvature decoding: { {k: round(v, 3) for k, v in curvature.items()} }", flush=True)

    # ---- AIT sheet category clustering (unpooled unit energies)
    rows = c5["stages"]["AIT"]["sheet"]
    e_units = maps_te["AIT"].pow(2).mean((2, 3))  # (N, units)
    cat_mean = torch.stack([e_units[y_te == c].mean(0) for c in range(len(CLASSES))])  # (10, units)
    z = (cat_mean - cat_mean.mean(0)) / (cat_mean.std(0) + 1e-8)
    pref = z.argmax(0)
    group_of = torch.zeros(len(CLASSES), dtype=torch.long)
    for gi, members in enumerate(SUPERORDINATE.values()):
        group_of[members] = gi
    pref_group = group_of[pref]
    dist = torus_distance(rows, rows)
    nbr = (dist == 1)

    def agreement(p):
        return float((p[:, None] == p[None, :])[nbr].float().mean())

    g2 = torch.Generator().manual_seed(123)
    null_c, null_g = [], []
    for _ in range(c5["checks"]["n_shuffles"]):
        perm = torch.randperm(len(pref), generator=g2)
        null_c.append(agreement(pref[perm]))
        null_g.append(agreement(pref_group[perm]))
    null_c, null_g = torch.tensor(null_c), torch.tensor(null_g)
    obs_c, obs_g = agreement(pref), agreement(pref_group)
    q = c5["checks"]["cluster_null_percentile"] / 100
    clustering = {
        "category_neighbor_agreement": obs_c, "category_null_mean": float(null_c.mean()),
        "category_null_p95": float(torch.quantile(null_c, q)), "category_p_value": float((null_c >= obs_c).float().mean()),
        "superordinate_neighbor_agreement": obs_g, "superordinate_null_mean": float(null_g.mean()),
        "superordinate_p_value": float((null_g >= obs_g).float().mean()),
        "units_preferring": {CLASSES[c]: int((pref == c).sum()) for c in range(len(CLASSES))},
    }
    print(f"AIT clustering: category agreement {obs_c:.3f} (null {null_c.mean():.3f}, p={clustering['category_p_value']:.4f}); "
          f"superordinate {obs_g:.3f} (null {null_g.mean():.3f}, p={clustering['superordinate_p_value']:.4f})", flush=True)

    ch = c5["checks"]
    checks = {
        "ait_decoding_above_chance": decoding["AIT"]["original"] >= ch["min_ait_accuracy"],
        "ait_category_clusters": obs_c > clustering["category_null_p95"],
    }

    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axes[0]
    xs = range(len(STAGE_ORDER))
    ax.plot(xs, [decoding[k]["original"] for k in STAGE_ORDER], color="#2a78d6", marker="o", linewidth=2,
            label="original")
    tcols = ["#eb6834", "#1baf7a", "#e87ba4"]
    for (tn, col) in zip(c5["transforms"], tcols):
        ax.plot(xs, [decoding[k][tn] for k in STAGE_ORDER], color=col, marker="o", linewidth=1.5, label=tn.replace("_", " "))
    ax.axhline(0.1, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xticks(list(xs), STAGE_ORDER)
    ax.set_ylim(0, 1)
    _style(ax, "Fashion-MNIST linear decoding by stage (decoder trained on originals)", "", "accuracy")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    ax = axes[1]
    ks = list(curvature)
    ax.bar(range(len(ks)), [curvature[k] for k in ks], color="#2a78d6", width=0.6)
    for i, k in enumerate(ks):
        ax.text(i, curvature[k], f"{curvature[k]:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.axhline(1 / len(cv["levels"]), color=MUTED, linestyle=":", linewidth=1)
    ax.set_xticks(range(len(ks)), ks)
    ax.set_ylim(0, 1)
    _style(ax, "Curvature magnitude (3 levels, any orientation)", "", "accuracy")
    fig.tight_layout()
    fig.savefig(out / "phase5_decoding.png", dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    grid = pref_group.view(rows, rows)
    colors = list(GROUP_COLORS.values())
    for r in range(rows):
        for c in range(rows):
            u = r * rows + c
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color=colors[int(grid[r, c])],
                                       alpha=float(0.35 + 0.65 * min(1.0, float(z[pref[u], u]) / 2.5))))
            ax.text(c + 0.5, r + 0.5, ABBR[int(pref[u])], ha="center", va="center", fontsize=6, color=INK)
    ax.set_xlim(0, rows)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("AIT sheet: preferred category (color = superordinate group)", fontsize=9, color=INK, loc="left")
    for gname, col in GROUP_COLORS.items():
        ax.plot([], [], "s", color=col, label=gname)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, labelcolor=INK)
    ax = axes[1]
    ax.hist(null_c.numpy(), bins=40, color="#c3c2b7", label="shuffled sheets (category)")
    ax.axvline(obs_c, color="#2a78d6", linewidth=2, label="AIT sheet (category)")
    ax.hist(null_g.numpy(), bins=40, color="#f2d9c8", label="shuffled sheets (superordinate)")
    ax.axvline(obs_g, color="#eb6834", linewidth=2, label="AIT sheet (superordinate)")
    _style(ax, "Neighbor agreement of preferred category", "fraction of sheet neighbors agreeing", "count")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "phase5_ait_sheet.png", dpi=120)

    report = {"decoding": decoding, "curvature_decoding": curvature, "ait_clustering": clustering,
              "whitening_floored": whitening_floored,
              "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"V2": v2.state_dict(), **{k: s.state_dict() for k, s in stages.items()}}, out / "phase5_stages.pt")
    print(json.dumps({"checks": checks}, indent=2))
    print(f"Phase 5 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
