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

Phase 5f (``configs/phase5f.yaml``): stages with ``bank: mix`` are
``LearnedHigherStage`` Gabor-mixing stages (Phase 5d/5e). Their mixing is
loaded (``load_bank``) or trained greedily on frame pairs propagated through
the stack below (``mix_training``) before the stage's whitening + TICA are
fit. ``refit_seeds`` repeats the matched readout over whitening + TICA refits
of every stage (filters fixed) and reports mean +- SE.

    python experiments/phase5_hierarchy.py [--config configs/phase5.yaml]
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
from gtv.data import CLASSES, REPO_PHOTOS, SUPERORDINATE, affine, load_fashion_mnist, load_grayscale, random_patches  # noqa: E402
from gtv.probes import curvature_fragment  # noqa: E402
from gtv.probes.decode import _standardize, fit_logistic, linear_decode  # noqa: E402
from gtv.stages import HigherStage, LearnedHigherStage, V2Stage, build_stage  # noqa: E402
from gtv.stages.tica import torus_distance  # noqa: E402
from gtv.temporal import fit_stage_filters  # noqa: E402

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

    stages, whitening_floored, mixing = {}, {}, {}
    m1_fit, s2_fit = low(x_fit)
    below = {"V1": m1_fit, "V2": s2_fit}
    prev_name, prev = "V2", s2_fit
    mt = c5.get("mix_training")
    if mt:  # frame pairs (image, transformed copy) for training mixing stages, propagated up the stack
        x_p, _ = load_fashion_mnist("train", mt["n_pairs"], size, seed=mt["pair_seed"])
        gp = torch.Generator().manual_seed(mt["pair_seed"])
        npr, rg = len(x_p), mt["range"]
        shift = (torch.rand(npr, 2, generator=gp) * 2 - 1) * rg["max_shift"]
        rot = (torch.rand(npr, generator=gp) * 2 - 1) * rg["max_rot_deg"]
        scl = torch.exp((torch.rand(npr, generator=gp) * 2 - 1) * rg["max_log_scale"])
        pair_maps = torch.stack([low(x_p)[1], low(affine(x_p, shift, rot, scl))[1]], 1)  # (N, 2, C, H, W)
    for name, sc in c5["stages"].items():
        skip_src = sc["skip"]
        kw = {"skip_channels": below[skip_src].shape[1] if skip_src else 0, "first_budget": sc.get("first_budget")}
        if sc.get("bank") == "mix":
            if skip_src:
                raise ValueError("mixing stages are trained without skips")
            st = LearnedHigherStage(name, prev.shape[1], sc["sheet"], sc["radius"], bank="mix",
                                    mix_radius=sc.get("mix_radius", 1), offset=sc.get("offset", 2), **kw)
            if sc.get("load_bank"):
                st.bank.weight.data.copy_(torch.load(ROOT / sc["load_bank"])["bank.weight"])
                mixing[name] = {"loaded_from": sc["load_bank"], "drift": st.bank.drift()}
            else:
                tk = mt["train"]
                hist = fit_stage_filters(st, pair_maps, None, temporal_weight=tk["temporal_weight"],
                                         tether=tk["tether"], white_weight=tk["white_weight"],
                                         n_steps=tk["n_steps"], lr=tk["lr"], batch_size=tk["batch_size"],
                                         refit_every=tk["refit_every"], refit_n=tk["refit_n"], border=sc["border"],
                                         seed=cfg["seed"], tica_kw=c5["tica"], optimizer=tk["optimizer"])
                mixing[name] = {"drift": hist["drift"][-1], "change_share": st.bank.offdiagonal_share()}
            print(f"{name} mixing: {mixing[name]}", flush=True)
        else:
            st = HigherStage(name, prev.shape[1], sc["sheet"], sc["radius"], **kw)
        skips = [below[skip_src]] if skip_src else None
        st.fit(prev, skips, border=sc["border"], seed=cfg["seed"], **c5["tica"])
        with torch.no_grad():
            nxt = torch.cat([st(pb, [s[i:i + 256] for s in skips] if skips else None)
                             for i, pb in zip(range(0, len(prev), 256), prev.split(256))])
        stages[name] = st
        below[name] = nxt
        if mt:
            with torch.no_grad():
                flat = pair_maps.flatten(0, 1)
                nxt_p = torch.cat([st(b) for b in flat.split(256)])
            pair_maps = nxt_p.view(npr, 2, *nxt_p.shape[1:])
        wh = st.tica.whitener
        floored = {"first_order": wh.parts[0].n_floored, "second_order": wh.parts[1].n_floored,
                   "joint": wh.joint.n_floored}
        prev_name, prev = name, nxt
        whitening_floored[name] = floored
        print(f"fitted {name}: {tuple(nxt.shape)}; whitening components floored {floored}", flush=True)

    def run(x, stack=None, lowmaps=None):
        """Per-stage maps for images x: V1 log energy; V2..AIT TICA outputs s."""
        stack = stack or stages
        m1, s2 = lowmaps if lowmaps is not None else low(x)
        maps = {"V1": m1, "V2": s2}
        prev = s2
        for name, sc in c5["stages"].items():
            skips = [maps[sc["skip"]]] if sc["skip"] else None
            with torch.no_grad():
                prev = torch.cat([stack[name](pb, [s[i:i + 256] for s in skips] if skips else None)
                                  for i, pb in zip(range(0, len(prev), 256), prev.split(256))])
            maps[name] = prev
        return maps

    def energy_of(name, s, stack=None):
        if name == "V1":
            return s
        st = v2 if name == "V2" else (stack or stages)[name]
        return st.pooled_energy(s)

    def readout(name, maps, x, stack=None):
        """2x2-pooled [s, pooled energy] (V1: log energy), flattened; pixels: the 28x28 content."""
        if name == "pixels":
            return F.adaptive_avg_pool2d(x, 28).flatten(1)
        s = maps[name]
        feat = s if name == "V1" else torch.cat([s, energy_of(name, s, stack)], 1)
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
    transformed, x_trans = {}, {}
    for tname, (sx, rot, scl) in c5["transforms"].items():
        with torch.no_grad():
            xt = affine(x_te, torch.tensor([[sx, 0.0]]).expand(n, 2), torch.full((n,), rot), torch.full((n,), scl))
        x_trans[tname] = xt
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

    # ---- fairer readout: the same PCA dimensionality for every stage, more training data
    # (feature counts grow up the hierarchy, so the plain readout overfits more at the top)
    mr = c5.get("matched_readout")
    matched, matched_seeds = {}, {}
    if mr:
        x_big, y_big = load_fashion_mnist("train", mr["n_train"], size, seed=7)
        maps_big = run(x_big)

        def matched_acc(ftr, tests):
            mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
            zt = (ftr - mu) / sd
            _, _, vh = torch.linalg.svd(zt - zt.mean(0), full_matrices=False)
            proj = vh[: min(mr["pca_dims"], vh.shape[0])].T
            zp = zt @ proj
            m2, s2 = zp.mean(0), zp.std(0) + 1e-6
            with torch.enable_grad():
                model = fit_logistic((zp - m2) / s2, y_big, len(CLASSES), l2=1e-3)
            with torch.no_grad():
                return {t: float((model(((((f - mu) / sd) @ proj) - m2) / s2).argmax(1) == y_te).float().mean())
                        for t, f in tests.items()}

        for k in STAGE_ORDER:
            tests = {"original": feats_te[k], **{tn: transformed[tn][k] for tn in transformed}}
            matched[k] = matched_acc(readout(k, maps_big, x_big), tests)
            print(f"matched readout {k}: {({t: round(v, 3) for t, v in matched[k].items()})}", flush=True)

        # the same readout after whitening + TICA refits of every upper stage (filters fixed)
        extra = [sd_ for sd_ in c5.get("refit_seeds", []) if sd_ != cfg["seed"]]
        if extra:
            if any(sc["skip"] for sc in c5["stages"].values()):
                raise ValueError("refit_seeds needs a stack without skips")
            upper = ["V2"] + list(c5["stages"])
            s2_cache = {"big": low(x_big)[1], "original": low(x_te)[1],
                        **{tn: low(xt_)[1] for tn, xt_ in x_trans.items()}}
            per_seed = {k: [{t: matched[k][t] for t in matched[k]}] for k in upper}
            for sd_ in extra:
                stack_s = copy.deepcopy(stages)
                prev_s = s2_fit
                for name, sc in c5["stages"].items():
                    stack_s[name].fit(prev_s, None, border=sc["border"], seed=sd_, **c5["tica"])
                    with torch.no_grad():
                        prev_s = torch.cat([stack_s[name](b) for b in prev_s.split(256)])
                mb = run(None, stack_s, (None, s2_cache["big"]))
                mt_ = {t: run(None, stack_s, (None, s2_cache[t])) for t in ["original"] + list(x_trans)}
                for k in upper:
                    tests = {t: readout(k, mt_[t], None, stack_s) for t in mt_}
                    per_seed[k].append(matched_acc(readout(k, mb, None, stack_s), tests))
                print(f"refit seed {sd_}: { {k: round(per_seed[k][-1]['original'], 3) for k in upper} }", flush=True)
            for k in upper:
                matched_seeds[k] = {}
                for t in per_seed[k][0]:
                    v = torch.tensor([p_[t] for p_ in per_seed[k]], dtype=torch.float64)
                    matched_seeds[k][t] = {"mean": float(v.mean()), "se": float(v.std() / math.sqrt(len(v)))}
                means = {t: round(m["mean"], 3) for t, m in matched_seeds[k].items()}
                print(f"matched readout {k}, {len(per_seed[k])} refits: {means}", flush=True)

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
    if matched and "max_ait_drop_vs_v2" in ch:
        mm = ({k: {t: m["mean"] for t, m in v.items()} for k, v in matched_seeds.items()} if matched_seeds
              else matched)  # refit-seed means when available
        checks["ait_no_decline_vs_v2_matched"] = mm["AIT"]["original"] >= mm["V2"]["original"] - ch["max_ait_drop_vs_v2"]
        checks["ait_rotation_at_least_v2_matched"] = mm["AIT"]["rotate_15deg"] >= mm["V2"]["rotate_15deg"]

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

    report = {"decoding": decoding, "matched_readout": matched, "matched_readout_refit_seeds": matched_seeds,
              "mixing": mixing, "curvature_decoding": curvature, "ait_clustering": clustering,
              "whitening_floored": whitening_floored,
              "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"V2": v2.state_dict(), **{k: s.state_dict() for k, s in stages.items()}}, out / "phase5_stages.pt")
    print(json.dumps({"checks": checks}, indent=2))
    print(f"Phase 5 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
