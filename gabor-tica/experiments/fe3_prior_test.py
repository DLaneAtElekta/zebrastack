"""Is the per-location TICA prior why converged settling destroys detection (FE-3)?

Same FE-2 model and FE-3 scenes. Converged (L-BFGS) settling with base
precision under four priors:
  tica_per_location   the FE-1/2/3 prior (units coupled only within a location)
  spatial_gaussian    stationary Gaussian over the whole latent map (full
                      cross-spectral matrix; the zero-frequency bin carries the
                      per-patch component), fitted to encoder latents on natural images
  tica_plus_spatial   both (product of experts)
  no_prior            accuracy only (control)
Two readouts: a linear readout of the latents (as FE-3), and the Phase 2
TICA readout of the *reconstructed* V1 maps (is the target information still
in the latents, just in other coordinates?).

    python experiments/fe3_prior_test.py [--config configs/fe3_prior_test.yaml]
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
from fe3_context_precision import _style, detection, make_scenes  # noqa: E402

from gtv.config import load_config  # noqa: E402
from gtv.data import REPO_PHOTOS, load_grayscale, random_patches  # noqa: E402
from gtv.generative import (  # noqa: E402
    AmortizedEncoder,
    ConvFactorAnalysis,
    NoPrior,
    Recognition,
    SpectralGaussianPrior,
    SumPrior,
    TICAPrior,
    fit_free_energy,
    settle_lbfgs,
)
from gtv.probes.sets import SCENE_KINDS  # noqa: E402
from gtv.stages import V2Stage, build_stage  # noqa: E402

INK, MUTED = "#1a1a19", "#6b6a63"
COLORS = {"amortized": "#86b6ef", "tica_per_location": "#eb6834", "spatial_gaussian": "#2a78d6",
          "tica_plus_spatial": "#1baf7a", "no_prior": "#a8a79d"}
LABELS = {"amortized": "encoder only", "tica_per_location": "TICA\nper-location",
          "spatial_gaussian": "spatial\nGaussian", "tica_plus_spatial": "TICA +\nspatial", "no_prior": "no prior"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "fe3_prior_test.yaml"))
    args = ap.parse_args()
    cp = load_config(args.config)
    c3f = load_config(ROOT / cp["base_config"])
    c2 = load_config(ROOT / c3f["base_config"])
    c1 = load_config(ROOT / c2["base_config"])
    c3 = load_config(ROOT / c1["base_config"])
    cfg = load_config(ROOT / c3["base_config"])
    vc, d, t, pc, sc = cfg["v2"], c3["data"], c1["train"], c2["precision"], c3f["scenes"]
    out = ROOT / cp["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["image_size"]
    gen = torch.Generator().manual_seed(cfg["seed"])

    # ---- the same FE-2 model as FE-3
    v1 = build_stage({"name": "V1", **cfg["v1"]})
    photos = load_grayscale(REPO_PHOTOS)
    train_photos, test_photos = photos[: d["train_photos"]], photos[d["train_photos"] :]
    with torch.no_grad():
        tr = v1(random_patches(train_photos, d["n_train"], size, gen))
        v1(random_patches(test_photos, d["n_test"], size, gen))  # keep the generator stream as in FE-3
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"], sheet=vc["sheet"],
                 radius=vc["radius"], dim=vc["dim"], tica_mode=vc["tica_mode"], **vc["design_B"])
    v2.fit(tr[: d["n_v2_fit"]], n_iter=vc["n_iter"], seed=cfg["seed"])
    tica = TICAPrior(v2.tica.h, v2.tica.eps)
    enc = AmortizedEncoder(v2, init="tica")
    dec = ConvFactorAnalysis(v2.tica.n_units, tr.shape[1], c3["decoder"]["kernel"])
    dec.fit_standardizer(tr)
    hist = fit_free_energy(tr, enc, dec, tica, torch.tensor(pc["init_log"]), n_iter=t["n_iter"], batch=t["batch"],
                           lr=t["lr"], warmup=t["warmup"], arrange_every=t["arrange_every"], learn_precision=True,
                           free_precision_after=pc["free_after"], log_precision_bounds=tuple(pc["bounds"]),
                           lr_precision=pc["lr"], generator=torch.Generator().manual_seed(cfg["seed"]))
    lp = hist["log_precision"]
    with torch.no_grad():
        mu_nat, _ = enc(tr)
    lat_size = tuple(mu_nat.shape[-2:])
    spatial = SpectralGaussianPrior(v2.tica.n_units, lat_size, cp["spectral_ridge"]).fit(mu_nat)
    priors = {"tica_per_location": tica, "spatial_gaussian": spatial,
              "tica_plus_spatial": SumPrior(tica, spatial), "no_prior": NoPrior()}

    with torch.no_grad():
        sv = {k: v1(make_scenes(test_photos, sc["n_per_kind"], k, size, sc, gen)) for k in SCENE_KINDS}
    tica_readout = Recognition(v2)  # the Phase 2 TICA stage, for reading reconstructed V1 maps

    def latent_feats(mu):
        pooled = torch.sqrt(torch.einsum("ij,bjhw->bihw", v2.tica.h, mu.pow(2)) + v2.tica.eps)
        return window_mean(torch.cat([mu, pooled], 1))

    def recon_feats(mu):
        with torch.no_grad():
            z = tica_readout.mean_from_features(tica_readout.features_batched(dec.to_v1(dec.mean(mu))))
        return latent_feats(z)

    mus = {"amortized": {k: enc(v)[0].detach() for k, v in sv.items()}}
    for name, prior in priors.items():
        mus[name] = {}
        for k, v in sv.items():
            x = dec.from_v1(v)
            mu0 = mus["amortized"][k]
            mus[name][k] = torch.cat([settle_lbfgs(xb, mb, dec, prior, lp, cp["converged_settle_steps"])
                                      for xb, mb in zip(x.split(40), mu0.split(40))])
        print(f"settled with {name}", flush=True)

    results = {}
    for name, m in mus.items():
        r_lat = detection({k: latent_feats(v) for k, v in m.items()}, c3f["hit_rate_for_fa"])
        r_rec = detection({k: recon_feats(v) for k, v in m.items()}, c3f["hit_rate_for_fa"])
        change = float((m["target"] - mus["amortized"]["target"]).norm() / mus["amortized"]["target"].norm())
        with torch.no_grad():
            spatial_energy = float(spatial.neg_log(m["target"]).mean())
            tica_energy = float(tica.neg_log(m["target"]).mean())
        results[name] = {"latent_readout": r_lat, "reconstruction_readout": r_rec, "latent_change": change,
                         "spatial_prior_energy": spatial_energy, "tica_prior_energy": tica_energy}
        print(f"{name}: latent d' {r_lat['dprime']:.2f} FA {r_lat['fa_distractor']:.2f} | via reconstruction "
              f"d' {r_rec['dprime']:.2f} FA {r_rec['fa_distractor']:.2f} | change {change:.2f}", flush=True)

    checks = {"spatial_prior_recovers_detection":
              results["spatial_gaussian"]["latent_readout"]["dprime"] >= cp["checks"]["min_dprime_spatial"]}

    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0), sharey=True)
    for ax, key, title in zip(axes, ("latent_readout", "reconstruction_readout"),
                              ("d′ from a linear readout of the latents", "d′ from the reconstructed V1 maps (Phase 2 readout)")):
        vals = [results[n][key]["dprime"] for n in names]
        ax.bar(range(len(names)), vals, color=[COLORS[n] for n in names], width=0.7)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(names)), [LABELS[n] for n in names], fontsize=7)
        ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
        _style(ax, title, "", "d′ (target vs absent)" if key == "latent_readout" else "")
    fig.suptitle("Converged settling under different latent priors (left of the dotted line: no settling)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "prior_test.png", dpi=120)

    report = {"results": results, "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"checks": checks}, indent=2))
    print(f"prior test {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
