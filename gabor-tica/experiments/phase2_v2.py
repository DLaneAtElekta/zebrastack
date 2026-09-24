"""Phase 2 exit check: V2 block, Design A (learned on V1 maps) vs. Design B
(second-order Gabors), both reduced by TICA on a topographic sheet.

Validation (plan Phase 2):
Readout features for A and B are both TICA outputs: signed coefficients s and
sheet-pooled energy. (Rectified-only |s| hides first-order information that is
carried by the sign; signed-only s hides second-order information in A.)

  * texture discrimination - first-order (orientation) and second-order
    (contrast-modulation orientation) texture classes, linear readout
  * texture-boundary detection - per-location linear detector, AUC
  * junction selectivity - line / L / T / X, linear readout
  * topography - neighbor vs. far energy correlation; orientation-preference map
  * sparsity - excess kurtosis of unit responses on natural images
Exit: B matches or beats A on texture and junction probes.

    python experiments/phase2_v2.py [--config configs/phase2.yaml]
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
    contrast_modulated_grating,
    excess_kurtosis,
    junction,
    linear_decode,
    linear_detect_auc,
    oriented_noise,
    texture_pair,
)
from gtv.probes.stimuli import JUNCTION_ARMS  # noqa: E402
from gtv.stages import V2Stage, build_stage  # noqa: E402
from gtv.stages.tica import torus_distance  # noqa: E402
from gtv.viz import fantasy_grid, sheet_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
REP_COLORS = {"V1": "#86b6ef", "A": "#eb6834", "B": "#2a78d6"}
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


def rand(gen, lo=0.0, hi=1.0):
    return lo + (hi - lo) * float(torch.rand(1, generator=gen))


# ---------------------------------------------------------------- probe sets
def texture_set(p, size, gen):
    """First-order: 8 orientation classes of band-pass noise."""
    imgs, labels = [], []
    for k in range(8):
        for _ in range(p["per_class"]):
            imgs.append(oriented_noise(size, k * math.pi / 8, p["texture_freq"], p["texture_bandwidth"], gen))
            labels.append(k)
    return torch.stack(imgs).unsqueeze(1), torch.tensor(labels)


def second_order_set(p, size, gen):
    """Second-order: 4 envelope-orientation classes; carrier orientation and
    envelope phase random, so mean V1 energy carries no class information."""
    imgs, labels = [], []
    for k in range(4):
        for _ in range(p["per_class"]):
            imgs.append(contrast_modulated_grating(
                size, p["cm_envelope_freq"], k * math.pi / 4,
                carrier_theta=rand(gen, 0, math.pi), carrier_freq=p["cm_carrier_freq"],
                generator=gen, envelope_phase=rand(gen, 0, 2 * math.pi)))
            labels.append(k)
    return torch.stack(imgs).unsqueeze(1), torch.tensor(labels)


def junction_set(p, size, gen):
    kinds = list(JUNCTION_ARMS)
    imgs, labels = [], []
    for k, kind in enumerate(kinds):
        for _ in range(p["per_class"]):
            c = (rand(gen, -2, 2), rand(gen, -2, 2))
            imgs.append(junction(size, kind, rand(gen, 0, 2 * math.pi), center=c))
            labels.append(k)
    return torch.stack(imgs).unsqueeze(1), torch.tensor(labels), kinds


def boundary_set(p, size, gen, cell=4, border=2):
    """Texture pairs with random orientations/boundary; per-cell labels:
    1 if the boundary passes within 3 px of the cell center, 0 if > 10 px."""
    imgs, dists = [], []
    grid = size // cell
    cc = (torch.arange(grid) + 0.5) * cell - size / 2
    cy, cx = torch.meshgrid(cc, cc, indexing="ij")
    for _ in range(p["boundary_images"]):
        ta = rand(gen, 0, math.pi)
        tb = ta + rand(gen, math.pi / 4, 3 * math.pi / 4)
        bt, off = rand(gen, 0, math.pi), rand(gen, -12, 12)
        img, _ = texture_pair(size, ta, tb, bt, p["texture_freq"], gen, off)
        imgs.append(img)
        dists.append((cx * math.cos(bt) + cy * math.sin(bt) - off).abs())
    d = torch.stack(dists)
    inner = torch.zeros(grid, grid, dtype=torch.bool)
    inner[border:-border, border:-border] = True
    return torch.stack(imgs).unsqueeze(1), d, inner


# ---------------------------------------------------------------- readouts
def window_mean(y, w):
    h = y.shape[2]
    lo = (h - w) // 2
    return y[:, :, lo : lo + w, lo : lo + w].mean((2, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase2.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    size, p, vc = cfg["image_size"], cfg["probes"], cfg["v2"]

    v1 = build_stage({"name": "V1", **cfg["v1"]})
    common = dict(sheet=vc["sheet"], radius=vc["radius"], dim=vc["dim"], tica_mode=vc["tica_mode"])
    stages = {
        "A": V2Stage(design="A", **common, **vc["design_A"]),
        "B": V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"],
                     **common, **vc["design_B"]),
    }

    # ---- fit TICA on natural images
    imgs = load_grayscale(REPO_PHOTOS)
    with torch.no_grad():
        nat = v1(random_patches(imgs, cfg["train"]["n_patches"], size, gen))
        nat_test = v1(random_patches(imgs, 128, size, gen))
    fit_kw = {"n_iter": vc["n_iter"], "seed": cfg["seed"]}
    if vc["tica_mode"] == "rica":
        fit_kw["lam"] = vc.get("rica_lam", 1.0)
    history = {k: s.fit(nat, **fit_kw) for k, s in stages.items()}

    def tica_out(st, m):
        """Both native TICA outputs: signed coefficients s (simple-cell-like)
        and sheet-pooled energy (complex-cell-like), stacked on channels."""
        s = st(m)
        return torch.cat([s, st.pooled_energy(s)], 1)

    def reps(x, image_level=True):
        """Representations of a probe batch: V1 (log energy), A and B (TICA outputs)."""
        with torch.no_grad():
            m = v1(x)
            r = {"V1": m, "A": tica_out(stages["A"], m), "B": tica_out(stages["B"], m)}
        if not image_level:
            return r
        w = p["window"]
        return {"V1": window_mean(r["V1"], 2 * w), "A": window_mean(r["A"], w), "B": window_mean(r["B"], w)}

    metrics, checks = {}, {}

    def acc(feats, y):
        return {k: sum(linear_decode(f, y, seed=s) for s in SEEDS) / len(SEEDS) for k, f in feats.items()}

    # ---- texture and junction decoding
    x1, y1 = texture_set(p, size, gen)
    x2, y2 = second_order_set(p, size, gen)
    xj, yj, kinds = junction_set(p, size, gen)
    metrics["texture_first_order_acc"] = acc(reps(x1), y1)
    metrics["texture_second_order_acc"] = acc(reps(x2), y2)
    metrics["junction_acc"] = acc(reps(xj), yj)

    # ---- boundary detection (per V2 cell; V1 pooled 2x to the same grid)
    xb, dist, inner = boundary_set(p, size, gen)
    rb = reps(xb, image_level=False)
    rb["V1"] = torch.nn.functional.avg_pool2d(rb["V1"], 2)
    pos, neg = (dist < 3) & inner, (dist > 10) & inner
    auc = {}
    for k, r in rb.items():
        f = r.permute(0, 2, 3, 1)  # (N, h, w, C)
        fp, fn = f[pos], f[neg]
        fn = fn[torch.randperm(len(fn), generator=gen)[: len(fp)]]  # balance classes
        xs = torch.cat([fp, fn])
        ys = torch.cat([torch.ones(len(fp)), torch.zeros(len(fn))])
        auc[k] = sum(linear_detect_auc(xs, ys, seed=s) for s in SEEDS) / len(SEEDS)
    metrics["boundary_auc"] = auc

    # ---- topography, sparsity, orientation maps
    rows = vc["sheet"]
    dsheet = torus_distance(rows, rows)
    off_diag = ~torch.eye(rows * rows, dtype=torch.bool)
    topo, sparsity, pref_maps = {}, {}, {}
    ori_resp = reps(x1)  # (N, C) mean |s| for each orientation texture
    for k, st in stages.items():
        with torch.no_grad():
            s = st(nat_test)[:, :, 2:-2, 2:-2]
        s = s.permute(0, 2, 3, 1).reshape(-1, s.shape[1])
        e = torch.corrcoef(s.pow(2).T)
        by_d = [float(e[(dsheet == d) & off_diag].mean()) for d in range(1, rows // 2 + 1)]
        topo[k] = {"energy_corr_by_sheet_distance": by_d, "ratio_near_far": by_d[0] / max(by_d[-1], 1e-6)}
        sparsity[k] = float(torch.tensor([excess_kurtosis(s[:, i]) for i in range(s.shape[1])]).median())
        # orientation preference via the doubled-angle vector average
        n_units = rows * rows
        # complex (pooled-energy) half of the TICA output
        mean_by_class = torch.stack([ori_resp[k][y1 == c, n_units:].mean(0) for c in range(8)])  # (8, n)
        ang = torch.arange(8) * 2 * math.pi / 8
        vec = (mean_by_class * torch.exp(1j * ang)[:, None]).sum(0) / mean_by_class.sum(0)
        pref = torch.remainder(torch.angle(vec) / 2, math.pi).view(rows, rows)
        sel = vec.abs().view(rows, rows)
        d_nb = [float(torch.remainder(pref - pref.roll(sh, dim), math.pi).sub(math.pi / 2).abs().neg().add(math.pi / 2).mean())
                for dim in (0, 1) for sh in (1, -1)]
        rnd = pref.flatten()[torch.randperm(rows * rows, generator=gen)].view(rows, rows)
        d_rnd = float(torch.remainder(pref - rnd, math.pi).sub(math.pi / 2).abs().neg().add(math.pi / 2).mean())
        pref_maps[k] = (pref, sel)
        # note: pooled energy averages over sheet neighbors, so some of this
        # smoothness is built in; ratio_near_far (on raw s) is the unbiased measure
        topo[k]["orientation_neighbor_diff_deg"] = math.degrees(sum(d_nb) / len(d_nb))
        topo[k]["orientation_random_diff_deg"] = math.degrees(d_rnd)
        topo[k]["median_orientation_selectivity"] = float(sel.median())
    metrics["topography"] = topo
    metrics["sparsity_excess_kurtosis_median"] = sparsity

    m = cfg["checks"]["margin"]
    for task in ("texture_first_order_acc", "texture_second_order_acc", "junction_acc", "boundary_auc"):
        checks[f"B_matches_A_{task}"] = metrics[task]["B"] >= metrics[task]["A"] - m
    checks["B_adds_second_order_over_V1"] = metrics["texture_second_order_acc"]["B"] > metrics["texture_second_order_acc"]["V1"] + 0.1
    checks["B_topographic"] = topo["B"]["ratio_near_far"] > cfg["checks"]["topography_ratio"]
    metrics["tica_loss"] = {k: [h[0], h[-1]] for k, h in history.items()}

    # ---------------------------------------------------------------- figures
    ex = torch.cat([x1[:: p["per_class"]][:4], x2[:: p["per_class"]], xj[:: p["per_class"]], xb[:4]])
    titles = [f"tex {k * 22.5:.0f}°" for k in range(4)] + [f"2nd-order {k * 45}°" for k in range(4)] + kinds + ["boundary"] * 4
    fantasy_grid(ex, ncols=8, titles=titles).savefig(out / "probes.png", dpi=120)

    tasks = [("texture_first_order_acc", "Texture\n(1st order)"), ("texture_second_order_acc", "Texture\n(2nd order)"),
             ("junction_acc", "Junctions"), ("boundary_auc", "Boundary\n(AUC)")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.26
    for i, rep in enumerate(("V1", "A", "B")):
        xs = [j + (i - 1) * width for j in range(len(tasks))]
        vals = [metrics[t][rep] for t, _ in tasks]
        ax.bar(xs, vals, width - 0.03, color=REP_COLORS[rep], label={"V1": "V1 only", "A": "Design A", "B": "Design B"}[rep])
        for x_, v in zip(xs, vals):
            ax.text(x_, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
    chance = [1 / 8, 1 / 4, 1 / 4, 0.5]
    for j, c in enumerate(chance):
        ax.hlines(c, j - 0.45, j + 0.45, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xticks(range(len(tasks)), [t for _, t in tasks])
    ax.set_ylim(0, 1.1)
    _style(ax, "Linear readout, held out (dotted = chance)", "", "accuracy / AUC")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(out / "ab_comparison.png", dpi=120)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for k in ("A", "B"):
        by_d = topo[k]["energy_corr_by_sheet_distance"]
        ax.plot(range(1, len(by_d) + 1), by_d, marker="o", markersize=5, linewidth=2, color=REP_COLORS[k], label=f"Design {k}")
    _style(ax, "Energy correlation vs. sheet distance", "distance on sheet (units)", "corr(s_i², s_j²)")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "topography_energy_corr.png", dpi=120)

    for k, (pref, sel) in pref_maps.items():
        sheet_map(pref, sel, title=f"Design {k}: preferred orientation").savefig(out / f"sheet_orientation_{k}.png", dpi=120)

    report = {"metrics": metrics, "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({k: s.state_dict() for k, s in stages.items()}, out / "v2_stages.pt")
    print(json.dumps(report, indent=2))
    print(f"Phase 2 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
