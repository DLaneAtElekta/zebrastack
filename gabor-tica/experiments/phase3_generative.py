"""Phase 3 exit check: generative path from V2 latents to V1 log-power maps.

  3a  convolutional factor analysis decoder, recognition fixed at the Phase 2
      Design B TICA; compare latent priors for fantasies
  3b  wake-sleep: wake updates decoder and prior, sleep updates recognition
  3c  dependent-variance prior (per-map gain + smooth variance fields)

Plausibility of generated V1 maps is measured by the statistic gap (mean
effect size of summary statistics vs held-out real maps; real-vs-real sets the
floor). Exit: stable wake-sleep and fantasies about as plausible as
reconstructions.

    python experiments/phase3_generative.py [--config configs/phase3.yaml]
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
    ConvFactorAnalysis,
    IIDGaussianPrior,
    Recognition,
    SpatialGaussianPrior,
    StationaryGaussianPrior,
    VarianceFieldPrior,
    fit_decoder,
    recon_r2,
    statistic_gap,
    wake_sleep,
)
from gtv.generative.diagnostics import STAT_GROUPS  # noqa: E402
from gtv.stages import V2Stage, build_stage  # noqa: E402
from gtv.viz import orientation_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
SERIES = {"iid": "#eb6834", "spatial": "#eda100", "stationary": "#2a78d6", "variance_3c": "#1baf7a",
          "recon_noise": "#6b6a63", "real_vs_real": "#c3c2b7"}
LABELS = {"iid": "fantasy, iid prior", "spatial": "fantasy, per-unit spatial prior",
          "stationary": "fantasy, global + local prior (3b)", "variance_3c": "fantasy, dependent-variance prior (3c)",
          "recon_noise": "reconstruction + noise", "real_vs_real": "real vs real (floor)"}


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


def sheet_ratio(z: torch.Tensor, v2: V2Stage) -> float:
    """Topography of latent maps: mean energy correlation of sheet neighbors
    divided by that of units >= half the sheet apart (as in Phase 2)."""
    from gtv.stages.tica import torus_distance

    rows = v2.tica.rows
    s = z[..., 2:-2, 2:-2].permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    e = torch.corrcoef(s.pow(2).T)
    dist = torus_distance(rows, rows)
    off = ~torch.eye(len(e), dtype=torch.bool)
    return float(e[(dist == 1) & off].mean() / e[(dist == rows // 2) & off].mean().clamp(min=1e-6))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase3.yaml"))
    args = ap.parse_args()
    c3 = load_config(args.config)
    cfg = load_config(ROOT / c3["base_config"])
    vc = cfg["v2"]
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    out = ROOT / c3["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    size, d = cfg["image_size"], c3["data"]

    # ---- V1 and the Phase 2 Design B stage, fitted on training photos only
    v1 = build_stage({"name": "V1", **cfg["v1"]})
    L, J = v1.bank.n_orientations, v1.bank.n_scales
    photos = load_grayscale(REPO_PHOTOS)
    with torch.no_grad():
        tr = v1(random_patches(photos[: d["train_photos"]], d["n_train"], size, gen))
        te = v1(random_patches(photos[d["train_photos"] :], d["n_test"], size, gen))
    v2 = V2Stage(design="B", v1_freqs=v1.bank.freq.tolist(), v1_decimate=cfg["v1"]["decimate"],
                 sheet=vc["sheet"], radius=vc["radius"], dim=vc["dim"], tica_mode=vc["tica_mode"], **vc["design_B"])
    v2.fit(tr[: d["n_v2_fit"]], n_iter=vc["n_iter"], seed=cfg["seed"])
    n_lat, lat_size = v2.tica.n_units, (tr.shape[-2] // 2, tr.shape[-1] // 2)
    nf = c3["n_fantasies"]
    metrics, checks = {}, {}

    def gap(fake):
        return statistic_gap(te, fake, L, J)

    def fantasies(dec, prior):
        with torch.no_grad():
            return dec.to_v1(dec.sample(prior.sample(nf, gen), gen))

    # ---- 3a: decoder with recognition fixed at TICA
    rec = Recognition(v2)
    rec_init = copy.deepcopy(rec)
    with torch.no_grad():
        z_tr = rec(tr)
    dc = c3["decoder"]
    dec = ConvFactorAnalysis(n_lat, tr.shape[1], dc["kernel"])
    dec.fit_standardizer(tr)
    fit_decoder(dec, dec.from_v1(tr), z_tr, n_iter=dc["n_iter"], lr=dc["lr"], generator=gen)
    dec.fit_noise_field(dec.from_v1(tr), z_tr)
    metrics["3a_test_r2"] = recon_r2(dec, rec, te)

    priors_3a = {
        "iid": IIDGaussianPrior(n_lat, lat_size),
        "spatial": SpatialGaussianPrior(n_lat, lat_size).fit(z_tr),
        "stationary": StationaryGaussianPrior(n_lat, lat_size, c3["prior"]["max_lag"]).fit(z_tr),
    }
    fant_3a = {k: fantasies(dec, p) for k, p in priors_3a.items()}
    with torch.no_grad():
        recon_noise = dec.to_v1(dec.sample(rec(te), gen))
    gaps = {f"3a_{k}": gap(v) for k, v in fant_3a.items()}
    gaps["recon_noise"] = gap(recon_noise)
    gaps["real_vs_real"] = gap(tr[: d["n_test"]])
    gaps["white_noise"] = gap(dec.to_v1(torch.randn(nf, *tr.shape[1:], generator=gen)))

    # ---- 3b: wake-sleep from the 3a decoder and the TICA recognition
    # (stabilized: fixed prior, q's noise initialized from its error on
    # fantasies, and a tether keeping recognition near its TICA start)
    ws = c3["wake_sleep"]
    dec_b, rec_b = copy.deepcopy(dec), copy.deepcopy(rec)
    prior_b = StationaryGaussianPrior(n_lat, lat_size, c3["prior"]["max_lag"])
    hist = wake_sleep(dec_b, rec_b, prior_b, tr, n_iter=ws["n_iter"], batch=ws["batch"], lr_wake=ws["lr_wake"],
                      lr_sleep=ws["lr_sleep"], refit_prior_every=ws["refit_prior_every"], tether=ws["tether"],
                      init_noise_from_sleep=ws["init_noise_from_sleep"], generator=gen)
    with torch.no_grad():
        z_b = rec_b(tr)
        zt_b, zt_0 = rec_b(te), rec_init(te)
    dec_b.fit_noise_field(dec_b.from_v1(tr), z_b)
    prior_b.fit(z_b)
    fant_b = fantasies(dec_b, prior_b)
    init_corr = torch.stack([torch.corrcoef(torch.stack([zt_b[:, i].flatten(), zt_0[:, i].flatten()]))[0, 1]
                             for i in range(n_lat)])
    metrics["3b_test_r2"] = recon_r2(dec_b, rec_b, te)
    metrics["3b_history"] = hist
    metrics["3b_latent_corr_with_tica_init"] = {"median": float(init_corr.median()), "min": float(init_corr.min())}
    metrics["sheet_near_far_energy_ratio"] = {"tica": sheet_ratio(zt_0, v2), "after_wake_sleep": sheet_ratio(zt_b, v2)}
    gaps["3b_stationary"] = gap(fant_b)

    # ---- 3c: dependent-variance prior on the 3b latents
    prior_c = VarianceFieldPrior(n_lat, lat_size).fit(z_b, generator=gen)
    fant_c = fantasies(dec_b, prior_c)
    gaps["3c_variance"] = gap(fant_c)
    metrics["3c_prior"] = {"per_map_gain_tau": prior_c.tau, "variance_field_rank": prior_c.rank,
                           "variance_field_width": prior_c.v_std}
    metrics["statistic_gap"] = gaps

    # ---- checks
    ch = c3["checks"]
    finite = all(torch.isfinite(p).all() for p in [*dec_b.parameters(), *rec_b.parameters()])
    checks["stable_finite"] = bool(finite)
    checks["stable_reconstruction"] = metrics["3b_test_r2"] >= metrics["3a_test_r2"] - ch["max_r2_drop"]
    checks["stable_no_collapse"] = (0.5 <= hist["latent_std"][-1] / hist["latent_std"][0] <= 2.0
                                    and metrics["3b_latent_corr_with_tica_init"]["median"] > ch["min_init_corr"])
    sr = metrics["sheet_near_far_energy_ratio"]
    checks["topography_kept"] = sr["after_wake_sleep"] >= ch["min_topography_kept"] * sr["tica"]
    checks["sleep_learns"] = hist["sleep_nll"][-1] < hist["sleep_nll"][0]
    checks["reconstruction_r2"] = metrics["3b_test_r2"] >= ch["min_test_r2"]
    best = min(gaps["3b_stationary"]["overall"], gaps["3c_variance"]["overall"])
    checks["fantasies_as_plausible_as_reconstructions"] = best <= ch["fantasy_vs_recon_ratio"] * gaps["recon_noise"]["overall"]
    checks["prior_structure_matters"] = gaps["3a_stationary"]["overall"] < gaps["3a_iid"]["overall"]

    # ---- figures
    orientation_grid({
        "real (held out)": te, "reconstruction": dec_b.to_v1(dec_b.mean(zt_b)).detach(),
        "fantasy, iid prior (3a)": fant_3a["iid"], "fantasy, global + local (3a)": fant_3a["stationary"],
        "fantasy, global + local (3b)": fant_b,
        "fantasy, dependent variance (3c)": fant_c,
    }, L, J, ncols=6).savefig(out / "fantasies_orientation.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    series = [("3a_iid", "iid"), ("3a_spatial", "spatial"), ("3b_stationary", "stationary"),
              ("3c_variance", "variance_3c"), ("recon_noise", "recon_noise"), ("real_vs_real", "real_vs_real")]
    groups = list(STAT_GROUPS)
    width = 0.8 / len(series)
    for i, (key, col) in enumerate(series):
        xs = [g + (i - (len(series) - 1) / 2) * width for g in range(len(groups))]
        ax.bar(xs, [gaps[key][g] for g in groups], width * 0.9, color=SERIES[col], label=LABELS[col])
    ax.set_xticks(range(len(groups)), [g.replace("_", "\n") for g in groups])
    ax.set_yscale("symlog", linthresh=1)
    _style(ax, "Statistic gap vs held-out real V1 maps (0 = matched; symlog scale)", "", "mean effect size")
    ax.grid(False, axis="x")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(out / "statistic_gap.png", dpi=120)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6))
    for ax, key, title in zip(axes, ("wake_nll", "sleep_nll", "latent_std"),
                              ("Wake: decoder NLL", "Sleep: recognition NLL", "Recognized latent std (median)")):
        ax.plot(hist["iter"], hist[key], color=SERIES["stationary"], linewidth=2, marker="o", markersize=4)
        _style(ax, title, "iteration")
    fig.tight_layout()
    fig.savefig(out / "wake_sleep_history.png", dpi=120)

    # projective fields: the V1 channels each latent drives (summed over space),
    # as an orientation x scale heatmap laid out on the TICA sheet
    with torch.no_grad():
        onehot = torch.zeros(n_lat, n_lat, *lat_size)
        onehot[torch.arange(n_lat), torch.arange(n_lat), lat_size[0] // 2, lat_size[1] // 2] = 1.0
        eff = (dec_b.mean(onehot) - dec_b.mean(torch.zeros(1, n_lat, *lat_size))).sum((2, 3))  # (n_lat, C)
    eff = eff.view(n_lat, J, L)
    rows = v2.tica.rows
    lim = float(eff.abs().max())
    fig, axes = plt.subplots(rows, rows, figsize=(1.0 * rows, 0.62 * rows))
    for i, ax in enumerate(axes.flat):
        ax.imshow(eff[i].numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Generative fields on the V2 sheet: V1 channels each latent drives\n"
                 "(each cell: rows = scale fine to coarse, columns = orientation 0-157°; red +, blue -)",
                 fontsize=8, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "projective_fields.png", dpi=130)
    # do sheet neighbors generate similar V1 patterns?
    from gtv.stages.tica import torus_distance

    ef = eff.flatten(1)
    cs = torch.corrcoef(ef)
    dist = torus_distance(rows, rows)
    off = ~torch.eye(n_lat, dtype=torch.bool)
    metrics["generative_field_similarity"] = {
        "sheet_neighbors": float(cs[(dist == 1) & off].abs().mean()),
        "far_apart": float(cs[(dist == rows // 2) & off].abs().mean()),
    }

    report = {"metrics": metrics, "checks": checks, "passed": all(checks.values())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"decoder": dec_b.state_dict(), "recognition": rec_b.state_dict(), "prior": prior_b.state_dict()},
               out / "generative.pt")
    summary = {k: v for k, v in metrics.items() if k != "3b_history"}
    summary["statistic_gap_overall"] = {k: round(v["overall"], 3) for k, v in gaps.items()}
    summary.pop("statistic_gap")
    print(json.dumps({"summary": summary, "checks": checks, "passed": report["passed"]}, indent=2))
    print(f"Phase 3 {'PASSED' if report['passed'] else 'FAILED'}; figures in {out}")


if __name__ == "__main__":
    main()
