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
3. Phase 8b (``glimpses`` in the config): border cells masked; a learned
   priority map (a linear per-cell sneaker detector on V4 features, trained on
   separate scenes of training items with known positions); serial glimpses
   (top-k locations with suppression), each routed window scored by an
   item-level classifier trained on routed training windows, scene score =
   max over glimpses.

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
from gtv.probes.decode import fit_logistic  # noqa: E402
from gtv.thalamus import (learned_priority, route_window, saliency_map, select_location, select_topk,  # noqa: E402
                          template_match_map)

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

    # ---- 3. Phase 8b: learned priority, border mask, serial glimpses
    gl = c8.get("glimpses")
    if gl:
        border = gl["border"]
        tr_items, tr_labels = items_tr, labels_tr
        gtr = torch.Generator().manual_seed(gl["train_seed"])

        def train_scene():
            cats = torch.randint(len(CLASSES), (tk["n_distractors"] + 1,), generator=gtr).tolist()
            cats[0] = tk["target"] if float(torch.rand(1, generator=gtr)) < 0.5 else cats[0]  # half contain a sneaker
            its = [tr_items[torch.nonzero(tr_labels == c).flatten()[
                int(torch.randint(int((tr_labels == c).sum()), (1,), generator=gtr))]] for c in cats]
            sc, ctr = clutter_scene(its, size, gtr, return_positions=True)
            return sc, ctr, cats

        made = [train_scene() for _ in range(gl["n_train_scenes"])]
        tr_imgs = torch.stack([m[0] for m in made]).unsqueeze(1)

        def scores_to_metrics(sc):
            """sc: {kind: (N,) scene scores}; all task scenes are held out."""
            sp, sa, sl_ = sc["present"], sc["absent"], sc["lookalike"]
            thr = torch.quantile(sp, 0.2)
            dp = lambda a, b: float((a.mean() - b.mean()) / (0.5 * (a.var() + b.var())).sqrt())  # noqa: E731
            return {"dprime": dp(sp, sa), "dprime_lookalike": dp(sp, sl_),
                    "fa_lookalike": float((sl_ > thr).float().mean()), "fa_absent": float((sa > thr).float().mean())}

        def fit_scorer(x, y):
            mu, sd = x.mean(0), x.std(0) + 1e-6
            with torch.enable_grad():
                model = fit_logistic((x - mu) / sd, y, 2, l2=1e-2)

            def score(f):
                with torch.no_grad():
                    o = model((f - mu) / sd)
                return o[:, 1] - o[:, 0]
            return score, model, mu, sd

        results["glimpses"] = {}
        for noise_label, T in (("noiseless", None), (f"noise_T{c8['response_noise_T']:g}", c8["response_noise_T"])):
            # learned per-cell detector on the (noisy) V4 features of the training scenes
            tf_, _ = run(tr_imgs, T, 500)
            h8 = tf_.shape[-1]
            pos = torch.zeros(len(made), h8, h8, dtype=torch.bool)
            for i, (_, ctr, cats) in enumerate(made):
                for (y, x), c in zip(ctr, cats):
                    if c == tk["target"]:
                        pos[i, min(int(y // cell), h8 - 1), min(int(x // cell), h8 - 1)] = True
            interior = torch.zeros(h8, h8, dtype=torch.bool)
            interior[border:h8 - border, border:h8 - border] = True
            cells = tf_.permute(0, 2, 3, 1)[:, interior]  # (N, n_int, C)
            labs = pos[:, interior]
            xp, xn = cells[labs], cells[~labs]
            xn = xn[torch.randperm(len(xn), generator=gtr)[: 3 * len(xp)]]
            _, det, dmu, dsd = fit_scorer(torch.cat([xp, xn]), torch.cat([torch.ones(len(xp)), torch.zeros(len(xn))]).long())
            w_det = (det.weight[1] - det.weight[0]).detach()
            b_det = float((det.bias[1] - det.bias[0]).detach())
            # item classifier on routed windows at every item of the training scenes
            ctr_all = torch.tensor([c for m in made for c in m[1]])
            cat_all = torch.tensor([c for m in made for c in m[2]])
            img_idx = torch.tensor([i for i, m in enumerate(made) for _ in m[1]])
            win = route_window(tr_imgs[img_idx], ctr_all, rt["window"], rt["scale"])
            wf = readout(run(win, T, 600)[1], "AIT")
            is_t = (cat_all == tk["target"]).long()
            informed_score, *_ = fit_scorer(wf, is_t)
            keep = ~torch.isin(cat_all, torch.tensor(tk["lookalikes"]))  # standard: never sees a lookalike
            standard_score, *_ = fit_scorer(wf[keep], is_t[keep])

            first = {k: run(scenes[k][0], T, i)[0] for i, k in enumerate(KINDS)}
            kmax = max(gl["k"])
            prio = {
                "learned": lambda f: learned_priority(f, w_det, b_det, dmu, dsd),
                "template": lambda f: template_match_map(f, templates, tk["target"]),
                "saliency": saliency_map,
            }
            gres = {}
            allp = list(prio) + ["random", "oracle"]
            for pname in [q for q in allp if q in (gl.get("priorities") or allp)]:
                per_kind, loc_hits = {}, None
                for i, k in enumerate(KINDS):
                    n = len(first[k])
                    if pname == "oracle":
                        cen = scenes[k][1].view(n, 1, 2)
                    elif pname == "random":
                        lo, hi = border * cell + cell / 2, size - border * cell - cell / 2
                        cen = lo + torch.rand(n, kmax, 2, generator=gr) * (hi - lo)
                    else:
                        cen = select_topk(prio[pname](first[k]), size, kmax, border)
                    kk = cen.shape[1]
                    win = route_window(scenes[k][0].repeat_interleave(kk, 0), cen.reshape(-1, 2), rt["window"], rt["scale"])
                    f = readout(run(win, T, 700 + i)[1], "AIT")
                    per_kind[k] = {"informed": informed_score(f).view(n, kk), "standard": standard_score(f).view(n, kk)}
                    if k == "present":
                        loc_hits = ((cen - scenes["present"][1].view(n, 1, 2)).norm(dim=2) <= rt["loc_tolerance"])
                ks = [1] if pname == "oracle" else gl["k"]
                for kg in ks:
                    r = {ro: scores_to_metrics({k: per_kind[k][ro][:, :kg].amax(1) for k in KINDS})
                         for ro in ("informed", "standard")}
                    r["localization_any"] = float(loc_hits[:, :kg].any(1).float().mean())
                    gres[f"{pname}_k{kg}"] = r
                    print(f"{noise_label} glimpses {pname} k={kg}: informed d'(vs lookalike) "
                          f"{r['informed']['dprime_lookalike']:.2f} FA(lookalike) {r['informed']['fa_lookalike']:.2f} "
                          f"d'(vs absent) {r['informed']['dprime']:.2f} | standard FA(lookalike) "
                          f"{r['standard']['fa_lookalike']:.2f} | target within any glimpse {r['localization_any']:.2f}",
                          flush=True)
            # Phase 8c: learned-priority variants (sub-cell refinement, item classifier trained on
            # the glimpses the gate actually selects in the training scenes)
            ctr_by_scene = [torch.tensor(m[1]) for m in made]
            cat_by_scene = [torch.tensor(m[2]) for m in made]
            glimpse_scorers, routed_task = {}, {}
            for v in gl.get("variants", []):
                ref = v["refine"]
                if v["classifier"] == "glimpse" and ref not in glimpse_scorers:
                    cen = select_topk(prio["learned"](tf_), size, kmax, border, refine=ref)  # (N, kmax, 2)
                    win = route_window(tr_imgs.repeat_interleave(kmax, 0), cen.reshape(-1, 2), rt["window"], rt["scale"])
                    gf = readout(run(win, T, 800)[1], "AIT")
                    near_t, near_l = [], []
                    for i in range(len(made)):
                        dist = (cen[i].view(-1, 1, 2) - ctr_by_scene[i].view(1, -1, 2)).norm(dim=2)  # (kmax, items)
                        close = dist <= rt["loc_tolerance"]
                        near_t.append((close & (cat_by_scene[i] == tk["target"]).view(1, -1)).any(1))
                        near_l.append((close & torch.isin(cat_by_scene[i], torch.tensor(tk["lookalikes"])).view(1, -1)).any(1))
                    near_t, near_l = torch.cat(near_t), torch.cat(near_l)
                    xg, yg = torch.cat([gf, wf]), torch.cat([near_t.long(), is_t])
                    keep_g = torch.cat([~(near_l & ~near_t), keep])
                    inf_s, *_ = fit_scorer(xg, yg)
                    std_s, *_ = fit_scorer(xg[keep_g], yg[keep_g])
                    glimpse_scorers[ref] = (inf_s, std_s)
                    print(f"{noise_label} glimpse-trained classifier (refine={ref}): {int(near_t.sum())} sneaker "
                          f"glimpses of {len(near_t)}", flush=True)
                if ref not in routed_task:
                    routed_task[ref] = {}
                    for i, k in enumerate(KINDS):
                        n = len(first[k])
                        cen = select_topk(prio["learned"](first[k]), size, kmax, border, refine=ref)
                        win = route_window(scenes[k][0].repeat_interleave(kmax, 0), cen.reshape(-1, 2),
                                           rt["window"], rt["scale"])
                        routed_task[ref][k] = (readout(run(win, T, 700 + i)[1], "AIT"), cen)
                inf_s, std_s = glimpse_scorers[ref] if v["classifier"] == "glimpse" else (informed_score, standard_score)
                n = len(first["present"])
                cen_p = routed_task[ref]["present"][1]
                hits = (cen_p - scenes["present"][1].view(n, 1, 2)).norm(dim=2) <= rt["loc_tolerance"]
                per = {k: {"informed": inf_s(routed_task[ref][k][0]).view(-1, kmax),
                           "standard": std_s(routed_task[ref][k][0]).view(-1, kmax)} for k in KINDS}
                for kg in gl["k"]:
                    r = {ro: scores_to_metrics({k: per[k][ro][:, :kg].amax(1) for k in KINDS})
                         for ro in ("informed", "standard")}
                    r["localization_any"] = float(hits[:, :kg].any(1).float().mean())
                    err = (cen_p[:, 0] - scenes["present"][1]).norm(dim=1)
                    r["first_glimpse_error_px_median"] = float(err[hits[:, 0]].median()) if hits[:, 0].any() else None
                    gres[f"learned[{v['name']}]_k{kg}"] = r
                    print(f"{noise_label} glimpses learned[{v['name']}] k={kg}: informed d'(vs lookalike) "
                          f"{r['informed']['dprime_lookalike']:.2f} FA(lookalike) {r['informed']['fa_lookalike']:.2f} "
                          f"d'(vs absent) {r['informed']['dprime']:.2f} | standard FA(lookalike) "
                          f"{r['standard']['fa_lookalike']:.2f} | target within any glimpse {r['localization_any']:.2f}"
                          + (f" | first-glimpse error {r['first_glimpse_error_px_median']:.1f} px"
                             if r["first_glimpse_error_px_median"] is not None else ""), flush=True)
            results["glimpses"][noise_label] = gres

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

    if gl:
        gn = results["glimpses"][f"noise_T{c8['response_noise_T']:g}"]
        routed = [k for k in gn if not k.startswith("oracle")]
        best = max(routed, key=lambda k: gn[k]["informed"]["dprime_lookalike"])
        checks["glimpses_raise_discrimination"] = (
            gn[best]["informed"]["dprime_lookalike"] - base_d >= ch["min_discrimination_gain"])
        checks["glimpses_lower_false_alarms"] = gn[best]["informed"]["fa_lookalike"] <= base_fa - ch["min_fa_drop"]
        loc_key = "learned_k1" if "learned_k1" in gn else next(k for k in gn if k.startswith("learned[") and k.endswith("_k1"))
        checks["learned_priority_localizes"] = gn[loc_key]["localization_any"] >= ch["min_localization"]
        if "min_oracle_gap_closed" in ch and "oracle_k1" in gn:
            # Phase 8c: fraction of the gap between pooled and oracle closed by the best routed condition
            gap = gn["oracle_k1"]["informed"]["dprime_lookalike"] - base_d
            closed = (gn[best]["informed"]["dprime_lookalike"] - base_d) / gap if gap > 0 else 0.0
            checks["closes_gap_to_oracle"] = closed >= ch["min_oracle_gap_closed"]
            results["oracle_gap_closed"] = closed
        results["glimpses_best"] = best
    report = {"results": results, "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"checks": checks}, indent=2))
    print(f"Phase 8 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
