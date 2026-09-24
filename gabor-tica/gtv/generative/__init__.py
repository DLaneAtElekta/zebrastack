"""Top-down generative path (Phase 3; Section 7 free-energy track later).

Modules:
    decoders.py     convolutional factor analysis, V2 latents -> V1 log-power maps
    priors.py       latent priors: iid, spatial Gaussian field, dependent variances (3c)
    wake_sleep.py   recognition model, decoder fitting (3a), wake-sleep loop (3b)
    diagnostics.py  V1-map statistics and real-vs-fantasy discriminability
"""

from .decoders import ConvFactorAnalysis
from .diagnostics import discriminability, map_statistics, statistic_gap
from .priors import IIDGaussianPrior, SpatialGaussianPrior, StationaryGaussianPrior, VarianceFieldPrior
from .wake_sleep import Recognition, fit_decoder, recon_r2, wake_sleep

__all__ = [
    "ConvFactorAnalysis",
    "IIDGaussianPrior",
    "Recognition",
    "SpatialGaussianPrior",
    "StationaryGaussianPrior",
    "VarianceFieldPrior",
    "discriminability",
    "fit_decoder",
    "map_statistics",
    "recon_r2",
    "statistic_gap",
    "wake_sleep",
]
