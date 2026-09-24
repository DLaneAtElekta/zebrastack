"""Is a set of V1 log-power maps plausible? Summary statistics plus a
real-vs-fantasy linear discriminator (0.5 = indistinguishable)."""

import torch

from gtv.probes.decode import linear_decode


STAT_GROUPS = ("mean", "log_std", "skew", "autocorr_lag1", "autocorr_lag4", "orientation_corr", "scale_corr")


def map_statistics(v1: torch.Tensor, n_orientations: int, n_scales: int, border: int = 4) -> torch.Tensor:
    """Per-map summary statistics (N, F) of V1 maps (N, L*J, H, W): per channel
    mean, log std, skewness, lag-1 and lag-4 spatial autocorrelation, and
    correlations with the neighboring orientation and the next coarser scale."""
    x = v1[..., border:-border, border:-border]
    n, c = x.shape[:2]
    mu = x.mean((2, 3), keepdim=True)
    sd = x.std((2, 3), keepdim=True).clamp(min=1e-6)
    z = (x - mu) / sd
    feats = [mu.flatten(1), sd.log().flatten(1), z.pow(3).mean((2, 3))]
    for lag in (1, 4):
        ax = (z[..., :, lag:] * z[..., :, :-lag]).mean((2, 3))
        ay = (z[..., lag:, :] * z[..., :-lag, :]).mean((2, 3))
        feats.append(0.5 * (ax + ay))
    zz = z.view(n, n_scales, n_orientations, *z.shape[2:])
    feats.append((zz * zz.roll(1, dims=2)).mean((3, 4)).flatten(1))
    if n_scales > 1:
        feats.append((zz[:, :-1] * zz[:, 1:]).mean((3, 4)).flatten(1))
    return torch.cat(feats, 1)


def discriminability(real: torch.Tensor, fake: torch.Tensor, n_orientations: int, n_scales: int,
                     seeds=(0, 1, 2), l2: float = 1e-2) -> float:
    """Held-out accuracy of a linear real-vs-fantasy classifier on map
    statistics; 0.5 means the statistics are indistinguishable."""
    fr = map_statistics(real, n_orientations, n_scales)
    ff = map_statistics(fake, n_orientations, n_scales)
    x = torch.cat([fr, ff])
    y = torch.cat([torch.zeros(len(fr)), torch.ones(len(ff))]).long()
    return sum(linear_decode(x, y, l2=l2, seed=s) for s in seeds) / len(seeds)


def _group_slices(n_orientations: int, n_scales: int) -> dict[str, slice]:
    c = n_orientations * n_scales
    sizes = [c] * 6 + ([(n_scales - 1) * n_orientations] if n_scales > 1 else [])
    out, start = {}, 0
    for name, size in zip(STAT_GROUPS, sizes):
        out[name] = slice(start, start + size)
        start += size
    return out


def statistic_gap(real: torch.Tensor, fake: torch.Tensor, n_orientations: int, n_scales: int) -> dict[str, float]:
    """Graded plausibility: for each summary statistic, |mean_fake - mean_real|
    in units of the real maps' spread (an effect size), averaged within each
    statistic group; ``overall`` averages the groups. 0 = matched."""
    fr = map_statistics(real, n_orientations, n_scales)
    ff = map_statistics(fake, n_orientations, n_scales)
    d = (ff.mean(0) - fr.mean(0)).abs() / fr.std(0).clamp(min=1e-6)
    gaps = {k: float(d[sl].mean()) for k, sl in _group_slices(n_orientations, n_scales).items()}
    gaps["overall"] = sum(gaps.values()) / len(gaps)
    return gaps
