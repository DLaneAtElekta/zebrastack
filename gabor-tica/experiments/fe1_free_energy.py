"""Section 7, step FE-1: replace wake-sleep with one variational free energy.

Encoder q(z | x) (fixed Design B front end + affine heads) and decoder
g(z) (convolutional FA) are trained together on real V1 maps by minimizing

    F = 1/2 pi ||x - g(z)||^2 - N/2 log pi + sum sqrt(h z^2 + eps) - H[q]

with pi fixed. Three runs: the literal Pi = I, pi from the Phase 3 residuals
with a TICA start, and the same from a random start (do TICA-like maps emerge?).

    python experiments/fe1_free_energy.py [--config configs/fe1.yaml]
"""

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from gtv.config import load_config  # noqa: E402
from gtv.data import REPO_PHOTOS, load_grayscale, random_patches  # noqa: E402
from gtv.generative import (  # noqa: E402
    AmortizedEncoder,
    ConvFactorAnalysis,
    Recognition,
    TICAPrior,
    active_units,
    encoder_r2,
    fit_decoder,
    fit_free_energy,
    recon_r2,
)
from gtv.probes import excess_kurtosis, linear_decode  # noqa: E402
from gtv.probes.sets import second_order_set  # noqa: E402
from gtv.stages import V2Stage, build_stage  # noqa: E402
from gtv.stages.tica import torus_distance  # noqa: E402
from gtv.viz import orientation_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
COLORS = {"identity_precision": "#eb6834", "tica_start": "#2a78d6", "random_start": "#1baf7a",
          "high_precision": "#e87ba4"}
LABELS = {"identity_precision": "Π = I, TICA start", "tica_start": "π ≈ 20, TICA start",
          "random_start": "π ≈ 20, random start", "high_precision": "π ≈ 150, TICA start"}
SEEDS = (0, 1, 2)


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def topography(mu: torch.Tensor, mask: torch.Tensor, rows: int) -> float:
    """Near/far energy-correlation ratio among active units: mean corr of
    (mu_i^2, mu_j^2) for sheet neighbors over that for units >= 3 apart."""
    s = mu[..., 2:-2, 2:-2].permute(0, 2, 3, 1).reshape(-1, mu.shape[1])
    e = torch.corrcoef(s.pow(2).T)
    d = torus_distance(rows, rows)
    m = mask[:, None] & mask[None, :] & ~torch.eye(len(mask), dtype=torch.bool)
    near, far = e[(d == 1) & m], e[(d >= 3) & m]
    if not len(near) or not len(far):
        return float("nan")
    return float(near.mean() / far.mean().clamp(min=1e-6))


def window_mean(y, w=4):
    lo = (y.shape[2] - w) // 2
    return y[:, :, lo : lo + w, lo : lo + w].mean((2, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "fe1.yaml"))
    args = ap.parse_args()
    cf = load_config(args.config)
    c3 = load_config(ROOT / cf["base_config"])
    cfg = load_config(ROOT / c3["base_config"])
    vc, d = cfg["v2"], c3["data"]
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    out = ROOT / cf["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["image_size"]

    # ---- same data, V1 and Phase 2 stage as Phase 3
    v1 = build_stage({"name": "V1", **cfg["v1"]})
    L, J = v1.bank.n_orientations, v1.bank.n_scales
    photos = load_grayscale(REPO_PHOTOS)
    with torch.no_grad():
        tr = v1(random_patches(photos[: d["train_photos"]], d["n_train"], size, gen))
        te = v1(random_patches(photos[d["train_photos"] :], d["n_test"], size, gen))
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"],
                 sheet=vc["sheet"], radius=vc["radius"], dim=vc["dim"], tica_mode=vc["tica_mode"], **vc["design_B"])
    v2.fit(tr[: d["n_v2_fit"]], n_iter=vc["n_iter"], seed=cfg["seed"])
    n_lat, rows = v2.tica.n_units, v2.tica.rows
    prior = TICAPrior(v2.tica.h, v2.tica.eps)

    # ---- Phase 3a reference: decoder fitted to the fixed TICA recognition
    rec = Recognition(v2)
    with torch.no_grad():
        z_tr = rec(tr)
    dec_3a = ConvFactorAnalysis(n_lat, tr.shape[1], c3["decoder"]["kernel"])
    dec_3a.fit_standardizer(tr)
    fit_decoder(dec_3a, dec_3a.from_v1(tr), z_tr, n_iter=c3["decoder"]["n_iter"], lr=c3["decoder"]["lr"], generator=gen)
    metrics = {"3a_test_r2": recon_r2(dec_3a, rec, te)}
    with torch.no_grad():
        mu_tica = rec(te)
    metrics["tica_topography"] = topography(mu_tica, torch.ones(n_lat, dtype=torch.bool), rows)

    # ---- probe sets for the second-order check
    probe_cfg = cfg["probes"]
    probe_sets = {f: second_order_set({**probe_cfg, "cm_envelope_freq": f}, size, gen)
                  for f in cf["probes"]["envelope_freqs"]}
    probe_feats = {f: rec.features_batched(v1(x)) for f, (x, _) in probe_sets.items()}

    def second_order_acc(mu_fn) -> dict:
        res = {}
        for f, (_, y) in probe_sets.items():
            with torch.no_grad():
                mu = mu_fn(probe_feats[f])
                pooled = torch.sqrt(torch.einsum("ij,bjhw->bihw", v2.tica.h, mu.pow(2)) + v2.tica.eps)
            feat = window_mean(torch.cat([mu, pooled], 1))
            res[str(f)] = sum(linear_decode(feat, y, seed=s) for s in SEEDS) / len(SEEDS)
        return res

    metrics["tica_second_order_acc"] = second_order_acc(rec.mean_from_features)

    # ---- FE-1 runs
    t = cf["train"]
    runs, hists, encs, decs = {}, {}, {}, {}
    for name, (init, log_pi) in cf["runs"].items():
        g = torch.Generator().manual_seed(cfg["seed"])
        enc = AmortizedEncoder(v2, init=init, seed=cfg["seed"])
        dec = ConvFactorAnalysis(n_lat, tr.shape[1], c3["decoder"]["kernel"])
        dec.fit_standardizer(tr)
        hist = fit_free_energy(tr, enc, dec, prior, torch.tensor(float(log_pi)), n_iter=t["n_iter"],
                               batch=t["batch"], lr=t["lr"], warmup=t["warmup"],
                               arrange_every=t["arrange_every"], generator=g, log_every=50)
        act = active_units(enc, te)
        with torch.no_grad():
            mu, log_sigma = enc(te)
        s = mu[..., 2:-2, 2:-2].permute(0, 2, 3, 1).reshape(-1, n_lat)
        runs[name] = {
            "init": init, "log_precision": log_pi,
            "test_r2": encoder_r2(enc, dec, te),
            "active_units": int(act.sum()),
            "topography_active": topography(mu, act, rows),
            "sparsity_kurtosis_active": float(torch.tensor([excess_kurtosis(s[:, i]) for i in act.nonzero().flatten()]).median()) if act.any() else float("nan"),
            "posterior_sigma_median": float(log_sigma.exp().median()),
            "F_per_element": [hist["F_per_element"][0], hist["F_per_element"][-1]],
            "second_order_acc": second_order_acc(lambda f, e=enc: e.from_features(f)[0]),
            # is the envelope in the latents at all? decode it from the reconstructed V1 maps
            "second_order_acc_via_reconstruction": second_order_acc(
                lambda f, e=enc, dd=dec: rec.mean_from_features(
                    rec.features_batched(dd.to_v1(dd.mean(e.from_features(f)[0]))))),
            "signal_to_noise": (mu.std((0, 2, 3)) / log_sigma.exp().mean((0, 2, 3))).tolist(),
        }
        hists[name], encs[name], decs[name] = hist, enc, dec
        print(f"{name}: R2 {runs[name]['test_r2']:.3f} active {runs[name]['active_units']} "
              f"topo {runs[name]['topography_active']:.2f}", flush=True)
    metrics["runs"] = runs

    # ---- checks (declared in the config before running)
    ch = cf["checks"]
    finite = all(torch.isfinite(p).all() for n in runs for p in [*encs[n].parameters(), *decs[n].parameters()])
    checks = {
        "no_divergence": bool(finite) and all(r["F_per_element"][1] < r["F_per_element"][0] for r in runs.values()),
        "reconstruction_matches_3a": max(runs["tica_start"]["test_r2"], runs["random_start"]["test_r2"])
        >= metrics["3a_test_r2"] - ch["max_r2_below_3a"],
        "topography_emerges_from_random_start": runs["random_start"]["topography_active"] > ch["min_emergent_topography"],
        "second_order_retained": all(v >= ch["min_second_order_acc"] for v in runs["tica_start"]["second_order_acc"].values()),
    }

    # ---- figures
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
    for ax, key, title in zip(axes, ("F_per_element", "accuracy", "prior", "entropy"),
                              ("Free energy F", "Accuracy term", "TICA prior term", "Entropy H[q]")):
        for name, h in hists.items():
            ax.plot(h["iter"], h[key], color=COLORS[name], linewidth=2, label=LABELS[name])
        ax.axvline(t["warmup"], color=MUTED, linewidth=0.8, linestyle=":")
        _style(ax, title, "iteration", "per V1 element" if key == "F_per_element" else "")
    axes[0].set_ylim(top=min(6, axes[0].get_ylim()[1]))
    axes[-1].legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "fe1_history.png", dpi=120)

    fig, axes = plt.subplots(1, len(runs), figsize=(3.3 * len(runs), 3.1))
    snr_max = max(max(r["signal_to_noise"]) for r in runs.values())
    for ax, (name, r) in zip(axes, runs.items()):
        im = ax.imshow(torch.tensor(r["signal_to_noise"]).view(rows, rows).numpy(), cmap="Blues", vmin=0, vmax=snr_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{LABELS[name]}\n{r['active_units']}/{n_lat} active", fontsize=8, color=INK, loc="left")
    fig.colorbar(im, ax=axes, fraction=0.02, label="std(μ) / σ")
    fig.suptitle("Posterior signal-to-noise per unit on the V2 sheet (collapsed units ≈ 0)", fontsize=9, color=INK,
                 x=0.01, ha="left")
    fig.savefig(out / "fe1_active_units.png", dpi=120, bbox_inches="tight")

    with torch.no_grad():
        rows_img = {"real (held out)": te}
        for name in runs:
            rows_img[f"reconstruction, {LABELS[name]}"] = decs[name].to_v1(decs[name].mean(encs[name](te)[0]))
    orientation_grid(rows_img, L, J, ncols=6).savefig(out / "fe1_reconstructions.png", dpi=120)

    report = {"metrics": metrics, "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({n: {"encoder": encs[n].state_dict(), "decoder": decs[n].state_dict()} for n in runs}, out / "fe1.pt")
    brief = {n: {k: v for k, v in r.items() if k != "signal_to_noise"} for n, r in runs.items()}
    print(json.dumps({"3a_test_r2": metrics["3a_test_r2"], "tica_topography": metrics["tica_topography"],
                      "tica_second_order_acc": metrics["tica_second_order_acc"], "runs": brief,
                      "checks": checks, "passed": report["passed"]}, indent=2))
    print(f"FE-1 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
