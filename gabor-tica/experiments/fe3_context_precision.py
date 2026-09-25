"""Section 7, step FE-3, as a V2-level analog of "cat d' in clutter".

A trained FE-2 model (encoder, decoder, learned base precision) sees scenes of
natural clutter with a texture patch at the center: target (45 deg contrast
modulation), distractor (135 deg), or absent (unmodulated). Conditions, with
no retraining of the model:
  amortized       encoder guess only
  settled_base    + settling on F with the base precision
  settled_fe3     + settling with the context precision fitted by minimizing F
                  on target-context scenes (FE-3 proper)
  settled_signal  + settling with expected-signal precision (comparator)
Detection d' (target vs absent) and false alarms (distractors scored as
targets) come from a linear readout of the posterior means.

    python experiments/fe3_context_precision.py [--config configs/fe3.yaml] [--calibrate]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from fe1_free_energy import window_mean  # noqa: E402

from gtv.config import load_config  # noqa: E402
from gtv.data import REPO_PHOTOS, load_grayscale, random_patches  # noqa: E402
from gtv.generative import (  # noqa: E402
    AmortizedEncoder,
    ConvFactorAnalysis,
    TICAPrior,
    context_precision_fe,
    encoder_r2,
    expected_signal_precision,
    fit_free_energy,
    settle,
    settle_lbfgs,
)
from gtv.probes.decode import _standardize, fit_logistic, train_test_split  # noqa: E402
from gtv.probes.sets import SCENE_KINDS, second_order_scene  # noqa: E402
from gtv.stages import V2Stage, build_stage  # noqa: E402

INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
COLORS = {"amortized": "#86b6ef", "settled_base": "#2a78d6", "settled_fe3": "#eb6834", "settled_signal": "#1baf7a",
          "converged_base": "#184f95", "converged_fe3": "#b54a1f", "converged_signal": "#11805a",
          "converged_fine_up": "#e87ba4", "converged_fine_down": "#a8a79d"}
LABELS = {"amortized": "amortized only", "settled_base": "settled, base",
          "settled_fe3": "settled, FE-3 context", "settled_signal": "settled, expected-signal",
          "converged_base": "converged, base", "converged_fe3": "converged, FE-3 context",
          "converged_signal": "converged, expected-signal", "converged_fine_up": "converged, fine +3",
          "converged_fine_down": "converged, fine −3"}
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


def make_scenes(photos, n, kind, size, sc, gen):
    bgs = random_patches(photos, n, size, gen)[:, 0]
    return torch.stack([second_order_scene(size, bgs[i], kind, gen, envelope_freq=sc["envelope_freq"],
                                           target_contrast=sc["target_contrast"],
                                           window_sigma=sc["window_sigma"]) for i in range(n)]).unsqueeze(1)


def detection(feats: dict, hit_rate: float) -> dict:
    """d' (target vs absent) and false-alarm rates from a held-out linear readout,
    averaged over train/test splits."""
    out = {"dprime": [], "fa_distractor": [], "fa_absent": []}
    n = len(feats["target"])
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        tr, te = train_test_split(n, 0.6, g)
        xtr = torch.cat([feats["target"][tr], feats["absent"][tr]])
        ytr = torch.cat([torch.ones(len(tr)), torch.zeros(len(tr))]).long()
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        with torch.enable_grad():
            model = fit_logistic((xtr - mu) / sd, ytr, 2, l2=1e-2)
        with torch.no_grad():
            score = {k: (lambda s: s[:, 1] - s[:, 0])(model((feats[k][te] - mu) / sd)) for k in SCENE_KINDS}
        st, sa = score["target"], score["absent"]
        out["dprime"].append(float((st.mean() - sa.mean()) / (0.5 * (st.var() + sa.var())).sqrt()))
        thr = torch.quantile(st, 1 - hit_rate)
        out["fa_distractor"].append(float((score["distractor"] > thr).float().mean()))
        out["fa_absent"].append(float((sa > thr).float().mean()))
    return {k: sum(v) / len(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "fe3.yaml"))
    ap.add_argument("--calibrate", action="store_true", help="only report baseline d' across target contrasts")
    args = ap.parse_args()
    c3f = load_config(args.config)
    c2 = load_config(ROOT / c3f["base_config"])
    c1 = load_config(ROOT / c2["base_config"])
    c3 = load_config(ROOT / c1["base_config"])
    cfg = load_config(ROOT / c3["base_config"])
    vc, d, t, pc, sc = cfg["v2"], c3["data"], c1["train"], c2["precision"], c3f["scenes"]
    out = ROOT / c3f["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["image_size"]
    gen = torch.Generator().manual_seed(cfg["seed"])

    # ---- the FE-2 model (normalized V1, learned precision), trained as in FE-2
    v1 = build_stage({"name": "V1", **cfg["v1"]})
    photos = load_grayscale(REPO_PHOTOS)
    train_photos, test_photos = photos[: d["train_photos"]], photos[d["train_photos"] :]
    with torch.no_grad():
        tr = v1(random_patches(train_photos, d["n_train"], size, gen))
        te = v1(random_patches(test_photos, d["n_test"], size, gen))
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"], sheet=vc["sheet"],
                 radius=vc["radius"], dim=vc["dim"], tica_mode=vc["tica_mode"], **vc["design_B"])
    v2.fit(tr[: d["n_v2_fit"]], n_iter=vc["n_iter"], seed=cfg["seed"])
    prior = TICAPrior(v2.tica.h, v2.tica.eps)
    enc = AmortizedEncoder(v2, init="tica")
    dec = ConvFactorAnalysis(v2.tica.n_units, tr.shape[1], c3["decoder"]["kernel"])
    dec.fit_standardizer(tr)
    hist = fit_free_energy(tr, enc, dec, prior, torch.tensor(pc["init_log"]), n_iter=t["n_iter"], batch=t["batch"],
                           lr=t["lr"], warmup=t["warmup"], arrange_every=t["arrange_every"], learn_precision=True,
                           free_precision_after=pc["free_after"], log_precision_bounds=tuple(pc["bounds"]),
                           lr_precision=pc["lr"], generator=torch.Generator().manual_seed(cfg["seed"]))
    base_lp = hist["log_precision"]
    st = c3f["settle"]

    def posterior(v1maps, lp, converged=False):
        x = dec.from_v1(v1maps)
        with torch.no_grad():
            mu0, _ = enc(v1maps)
        if lp is None:
            return mu0
        if converged:
            return torch.cat([settle_lbfgs(xb, mb, dec, prior, lp, c3f["converged_settle_steps"])
                              for xb, mb in zip(x.split(40), mu0.split(40))])
        return torch.cat([settle(xb, mb, dec, prior, lp, st["n_steps"], st["step"])
                          for xb, mb in zip(x.split(64), mu0.split(64))])

    def features(mu):
        pooled = torch.sqrt(torch.einsum("ij,bjhw->bihw", v2.tica.h, mu.pow(2)) + v2.tica.eps)
        return window_mean(torch.cat([mu, pooled], 1))

    if args.calibrate:
        for contrast in (0.5, 1.0, 1.5, 2.0, 3.0):
            scc = {**sc, "target_contrast": contrast}
            with torch.no_grad():
                sv = {k: v1(make_scenes(test_photos, sc["n_per_kind"], k, size, scc, gen)) for k in SCENE_KINDS}
            res = detection({k: features(posterior(v, None)) for k, v in sv.items()}, c3f["hit_rate_for_fa"])
            print(f"contrast {contrast}: amortized d' {res['dprime']:.2f}  FA(distractor) {res['fa_distractor']:.2f}", flush=True)
        return

    # ---- scenes: held-out photos for testing; training photos for fitting context precision
    with torch.no_grad():
        scenes = {k: make_scenes(test_photos, sc["n_per_kind"], k, size, sc, gen) for k in SCENE_KINDS}
        sv = {k: v1(v) for k, v in scenes.items()}
        ctx_target = v1(make_scenes(train_photos, sc["n_context"], "target", size, sc, gen))
        ctx_absent = v1(make_scenes(train_photos, sc["n_context"], "absent", size, sc, gen))

    fe3_lp = context_precision_fe(ctx_target, enc, dec, prior, base_lp, n_rounds=c3f["context_fit_rounds"],
                                  n_steps=st["n_steps"], step=st["step"])
    lo = sv["target"].shape[-1] // 2 - 8
    signal_lp = expected_signal_precision(ctx_target, ctx_absent, dec, base_lp, slice(lo, lo + 16))
    L, J = v1.bank.n_orientations, v1.bank.n_scales
    fine = torch.zeros_like(base_lp)
    fine[:L] = 3.0  # channel order is (scale, orientation); scale 0 is the finest
    precisions = {"amortized": None, "settled_base": base_lp, "settled_fe3": fe3_lp, "settled_signal": signal_lp,
                  "converged_base": base_lp, "converged_fe3": fe3_lp, "converged_signal": signal_lp,
                  "converged_fine_up": base_lp + fine, "converged_fine_down": base_lp - fine}

    results = {}
    for name, lp in precisions.items():
        conv = name.startswith("converged")
        mus = {k: posterior(v, lp, conv) for k, v in sv.items()}
        results[name] = detection({k: features(m) for k, m in mus.items()}, c3f["hit_rate_for_fa"])
        with torch.no_grad():
            mu0, _ = enc(sv["target"])
        results[name]["latent_change_from_amortized"] = float((mus["target"] - mu0).norm() / mu0.norm())
        results[name]["natural_r2"] = encoder_r2(enc, dec, te) if lp is None else float(
            1 - ((dec.from_v1(te) - dec.mean(posterior(te, lp, conv)))[..., 4:-4, 4:-4].pow(2).mean()
                 / dec.from_v1(te)[..., 4:-4, 4:-4].var()))
        print(f"{name}: d' {results[name]['dprime']:.2f}  FA(distractor) {results[name]['fa_distractor']:.2f}  "
              f"FA(absent) {results[name]['fa_absent']:.2f}", flush=True)

    prof = {"base": base_lp.view(J, L), "fe3_context": fe3_lp.view(J, L), "expected_signal": signal_lp.view(J, L)}
    ch = c3f["checks"]
    checks = {
        "fe3_raises_dprime": results["settled_fe3"]["dprime"] - results["settled_base"]["dprime"] >= ch["min_dprime_gain"],
        "fe3_no_extra_false_alarms": results["settled_fe3"]["fa_distractor"] - results["settled_base"]["fa_distractor"]
        <= ch["max_fa_increase"],
    }
    findings = {
        "fe3_delta_log_precision_by_scale": (prof["fe3_context"] - prof["base"]).mean(1).tolist(),
        "signal_delta_log_precision_by_scale": (prof["expected_signal"] - prof["base"]).mean(1).tolist(),
        "signal_dprime_gain": results["settled_signal"]["dprime"] - results["settled_base"]["dprime"],
    }

    # ---- figures
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0))
    names = list(results)
    for ax, key, title in zip(axes, ("dprime", "fa_distractor"),
                              ("Detection d′ (target vs absent)", "False alarms on distractors (hit rate 0.8)")):
        vals = [results[n][key] for n in names]
        ax.bar(range(len(names)), vals, color=[COLORS[n] for n in names], width=0.7)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(names)), [LABELS[n].replace(", ", ",\n") for n in names], fontsize=6)
        ax.axvline(3.5, color=MUTED, linewidth=0.8, linestyle=":")
        _style(ax, title)
    fig.tight_layout()
    fig.savefig(out / "fe3_detection.png", dpi=120)

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.4))
    deltas = {"base": prof["base"], "FE-3 context − base": prof["fe3_context"] - prof["base"],
              "expected-signal − base": prof["expected_signal"] - prof["base"]}
    for ax, (title, m) in zip(axes, deltas.items()):
        lim = float(m.abs().max()) or 1.0
        cmap, vmin = ("Blues", float(m.min())) if title == "base" else ("RdBu_r", -lim)
        im = ax.imshow(m.numpy(), cmap=cmap, vmin=vmin, vmax=float(m.max()) if title == "base" else lim, aspect="auto")
        for j in range(J):
            for l in range(L):
                ax.text(l, j, f"{float(m[j, l]):+.1f}" if title != "base" else f"{float(m[j, l]):.1f}",
                        ha="center", va="center", fontsize=6, color=INK)
        ax.set_xticks(range(L), [f"{int(180 * l / L)}°" for l in range(L)], fontsize=6)
        ax.set_yticks(range(J), ["fine", "mid", "coarse"][:J], fontsize=7)
        ax.set_title(f"log π: {title}", fontsize=9, color=INK, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out / "fe3_precision_profiles.png", dpi=120)

    report = {"results": results, "checks": checks, "findings": findings,
              "precision_profiles": {k: v.tolist() for k, v in prof.items()}, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"results": results, "checks": checks, "findings": findings, "passed": report["passed"]}, indent=2))
    print(f"FE-3 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
