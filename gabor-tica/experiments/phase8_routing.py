"""Phase 8: pulvinar-style routing (object-centered processing) in clutter.

Phases 6-7 found that in 4-object scenes the stack pools the objects together
by AIT (sneaker-vs-lookalike d' 0.53 even without noise), so attention and
expectation at V4 have little to act on. Here a shifter-style gate selects one
location from a priority map on V4's 8 x 8 sheet, cuts a window around it
from the scene, rescales it to the stack's training scale and centers it
(``gtv.thalamus.routing``), and the stack is run again on that window.

1. Diagnostic: sneaker-vs-lookalike discrimination at V4 and AIT for single
   objects vs 4-object scenes (pooled, no routing), noiseless.
2. Routing, noiseless and with the Phase 6c bottleneck (noise at V4, T):
   pooled (no routing, the Phase 7 reference); template routing (priority =
   match to the sneaker template, from the noisy V4 signal when noise is on);
   saliency routing (V4 feature energy); random location; oracle (the true
   center of the task item). Readouts as Phase 7c: standard (present vs
   absent) and informed (lookalikes among the negatives), on held-out scenes.
   Localization: routed center within ``loc_tolerance`` px of the target.

    python experiments/phase8_routing.py [--config configs/phase8.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gtv.config import load_config  # noqa: E402
from gtv.data import CLASSES, load_fashion_mnist  # noqa: E402
from gtv.probes.sets import clutter_scene  # noqa: E402
from gtv.stages import V2Stage, build_higher_stack, build_stage  # noqa: E402
from gtv.thalamus import route_window, saliency_map, select_location, template_match_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from phase6_attention import KINDS, detection  # noqa: E402

INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
INF = ("absent", "lookalike")


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, axis="y")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase8.yaml"))
    args = ap.parse_args()
    c8 = load_config(args.config)
    c5 = load_config(ROOT / c8["base_config"])
    cfg = load_config(ROOT / c5["base_config"])
    vc, tk, rt = cfg["v2"], c8["task"], c8["routing"]
    out = ROOT / c8["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["image_size"]
    gen = torch.Generator().manual_seed(cfg["seed"])

    # ---- the Phase 5b stack, loaded
    v1 = build_stage({"name": "V1", **cfg["v1"]})
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"], sheet=vc["sheet"],
                 radius=vc["radius"], dim=vc["dim"], tica_mode=vc["tica_mode"], **vc["design_B"])
    ck = torch.load(ROOT / c8["stack_checkpoint"])
    v2.load_state_dict(ck["V2"])
    stack = build_higher_stack(c5["stages"], {"V1": v1.bank.n_filters, "V2": v2.tica.n_units})
    for name, st in stack.items():
        st.load_state_dict(ck[name])
    names = list(stack)
    v4 = stack["V4"]
    sl = v4.second_order_slice

    def run(images, noise_T=None, noise_seed=0):
        """-> (V4 second-order features (B, n_second, 8, 8), {stage: TICA outputs})."""
        ngen = torch.Generator().manual_seed(noise_seed)
        feats, maps = [], {n: [] for n in names}
        with torch.no_grad():
            for xb in images.split(128):
                s2 = v2(v1(xb))
                f = v4.features(s2, noise_T=noise_T, generator=ngen)
                feats.append(f[:, sl])
                m, b = v4.tica.affine
                prev = torch.einsum("nd,bdhw->bnhw", m, f) + b.view(1, -1, 1, 1)
                maps["V4"].append(prev)
                for n in names[1:]:
                    prev = stack[n](prev)
                    maps[n].append(prev)
        return torch.cat(feats), {n: torch.cat(v) for n, v in maps.items()}

    def readout(maps, name):
        s = maps[name]
        return F.adaptive_avg_pool2d(torch.cat([s, stack[name].pooled_energy(s)], 1), 2).flatten(1)

    # ---- templates at scene scale: V4 features at the item's cell in single-item scenes
    items_tr, labels_tr = load_fashion_mnist("train", None, raw=True)
    tpl_scenes, tpl_centers, tpl_labels = [], [], []
    gt = torch.Generator().manual_seed(5)
    for c in range(len(CLASSES)):
        idx = torch.nonzero(labels_tr == c).flatten()
        for i in idx[torch.randperm(len(idx), generator=gt)[: c8["templates_per_class"]]].tolist():
            sc, ctr = clutter_scene([items_tr[i]], size, gt, return_positions=True)
            tpl_scenes.append(sc)
            tpl_centers.append(ctr[0])
            tpl_labels.append(c)
    tf, _ = run(torch.stack(tpl_scenes).unsqueeze(1))
    cell = size / tf.shape[-1]
    cy = torch.tensor([int(y // cell) for y, _ in tpl_centers])
    cx = torch.tensor([int(x // cell) for _, x in tpl_centers])
    at_item = tf[torch.arange(len(tf)), :, cy, cx]  # (N, n_second)
    tl = torch.tensor(tpl_labels)
    templates = torch.stack([at_item[tl == c].mean(0) for c in range(len(CLASSES))])
    print(f"templates from {len(tl)} single-item scenes", flush=True)

    # ---- scenes (test items), with the task item's center
    items, labels = load_fashion_mnist("test", None, raw=True)
    pool = {c: items[labels == c] for c in range(len(CLASSES))}
    others = [c for c in range(len(CLASSES)) if c != tk["target"] and c not in tk["lookalikes"]]

    def pick(c):
        return pool[c][int(torch.randint(len(pool[c]), (1,), generator=gen))]

    def make(kind, n_dis):
        dis = [pick(others[int(torch.randint(len(others), (1,), generator=gen))]) for _ in range(n_dis)]
        if kind == "present":
            first = pick(tk["target"])
        elif kind == "lookalike":
            first = pick(tk["lookalikes"][int(torch.randint(len(tk["lookalikes"]), (1,), generator=gen))])
        else:
            first = pick(others[int(torch.randint(len(others), (1,), generator=gen))])
        sc, ctr = clutter_scene([first] + dis, size, gen, return_positions=True)
        return sc, ctr[0]

    def scene_set(n_dis):
        sets = {}
        for k in KINDS:
            made = [make(k, n_dis) for _ in range(tk["n_per_kind"])]
            sets[k] = (torch.stack([m[0] for m in made]).unsqueeze(1), torch.tensor([m[1] for m in made]))
        return sets

    def score(feats_by_kind):
        std = detection(feats_by_kind)
        inf = detection(feats_by_kind, negatives=INF)
        return {"standard": std, "informed": inf}

    results = {"diagnostic": {}, "routing": {}}

    # ---- 1. diagnostic: single objects vs clutter, pooled, noiseless
    scenes_by_n = {}
    for n_dis in c8["diagnostic_distractors"]:
        scenes_by_n[n_dis] = scene_set(n_dis)
        maps = {k: run(scenes_by_n[n_dis][k][0])[1] for k in KINDS}
        results["diagnostic"][n_dis] = {st: score({k: readout(maps[k], st) for k in KINDS}) for st in ("V4", "AIT")}
        d = results["diagnostic"][n_dis]
        print(f"diagnostic, {n_dis} distractors: sneaker-vs-lookalike d' (informed) V4 "
              f"{d['V4']['informed']['dprime_lookalike']:.2f}, AIT {d['AIT']['informed']['dprime_lookalike']:.2f}; "
              f"present-vs-absent d' AIT {d['AIT']['standard']['dprime']:.2f}", flush=True)

    # ---- 2. routing in the task scenes
    scenes = scenes_by_n.get(tk["n_distractors"]) or scene_set(tk["n_distractors"])
    gr = torch.Generator().manual_seed(99)
    for noise_label, T in (("noiseless", None), (f"noise_T{c8['response_noise_T']:g}", c8["response_noise_T"])):
        res = {}
        first = {k: run(scenes[k][0], T, i) for i, k in enumerate(KINDS)}  # first pass (decides where to route)
        res["pooled"] = score({k: readout(first[k][1], "AIT") for k in KINDS})
        centers = {}
        for k in KINDS:
            f = first[k][0]
            n = len(f)
            centers[k] = {
                "template": select_location(template_match_map(f, templates, tk["target"]), size),
                "saliency": select_location(saliency_map(f), size),
                "random": torch.randint(rt["window"] // 2, size - rt["window"] // 2 + 1, (n, 2), generator=gr).float(),
                "oracle": scenes[k][1],
            }
        loc = {}
        for mode in ("template", "saliency", "random", "oracle"):
            routed = {}
            for i, k in enumerate(KINDS):
                img = route_window(scenes[k][0], centers[k][mode], rt["window"], rt["scale"])
                routed[k] = readout(run(img, T, 100 + i)[1], "AIT")
            res[f"routed_{mode}"] = score(routed)
            dist = (centers["present"][mode] - scenes["present"][1]).norm(dim=1)
            loc[mode] = float((dist <= rt["loc_tolerance"]).float().mean())
            res[f"routed_{mode}"]["localization"] = loc[mode]
        results["routing"][noise_label] = res
        for cname, r in res.items():
            print(f"{noise_label} {cname}: present-vs-absent d' {r['standard']['dprime']:.2f} "
                  f"FA(lookalike) {r['standard']['fa_lookalike']:.2f} | informed: FA {r['informed']['fa_lookalike']:.2f} "
                  f"d'(vs lookalike) {r['informed']['dprime_lookalike']:.2f}"
                  + (f" | localization {r['localization']:.2f}" if "localization" in r else ""), flush=True)

    # ---- checks (declared before running), on the noisy condition
    ch = c8["checks"]
    rn = results["routing"][f"noise_T{c8['response_noise_T']:g}"]
    base_d = rn["pooled"]["informed"]["dprime_lookalike"]
    base_fa = rn["pooled"]["informed"]["fa_lookalike"]
    rtpl = rn["routed_template"]
    checks = {
        "routing_raises_discrimination": rtpl["informed"]["dprime_lookalike"] - base_d >= ch["min_discrimination_gain"],
        "routing_lowers_false_alarms": rtpl["informed"]["fa_lookalike"] <= base_fa - ch["min_fa_drop"],
        "routing_localizes_target": rtpl["localization"] >= ch["min_localization"],
        "template_beats_saliency": rtpl["informed"]["dprime_lookalike"] > rn["routed_saliency"]["informed"]["dprime_lookalike"],
    }

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    modes = ["pooled", "routed_random", "routed_saliency", "routed_template", "routed_oracle"]
    cols = ["#c3c2b7", "#e4c9a8", "#eb6834", "#2a78d6", "#1baf7a"]
    for ax, (nl, rr) in zip(axes[:2], results["routing"].items()):
        vals = [rr[m]["informed"]["dprime_lookalike"] for m in modes]
        ax.bar(range(len(modes)), vals, color=cols, width=0.65)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(modes)), [m.replace("routed_", "routed\n") for m in modes], fontsize=7)
        _style(ax, f"Sneaker vs lookalike d' at AIT, informed readout ({nl})", "", "d'")
    ax = axes[2]
    ex = scenes["present"][0][:4, 0]
    routed_ex = route_window(scenes["present"][0][:4], centers["present"]["template"][:4], rt["window"], rt["scale"])
    grid = torch.cat([torch.cat(list(ex), 1), torch.cat(list(routed_ex[:, 0]), 1)], 0)
    ax.imshow(grid.numpy(), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([size / 2, size * 1.5], ["scene", "template-\nrouted"], fontsize=7)
    ax.set_title("Sneaker-present scenes and their routed windows (noisy)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "phase8_routing.png", dpi=120)

    report = {"results": results, "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"checks": checks}, indent=2))
    print(f"Phase 8 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
