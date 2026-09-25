"""Top-down generative path (Phase 3; Section 7 free-energy track later).

Modules:
    decoders.py     convolutional factor analysis, V2 latents -> V1 log-power maps
    priors.py       latent priors: iid, spatial Gaussian field, dependent variances (3c)
    wake_sleep.py   recognition model, decoder fitting (3a), wake-sleep loop (3b)
    free_energy.py  Section 7 FE-1/FE-2: one free-energy objective, amortized TICA-prior VAE
    context.py      Section 7 FE-3: settling on F and context-conditioned precision
    diagnostics.py  V1-map statistics and real-vs-fantasy discriminability
"""

from .context import context_precision_fe, expected_signal_precision, residual_variance, settle, settle_lbfgs
from .decoders import ConvFactorAnalysis
from .free_energy import (
    AmortizedEncoder,
    TICAPrior,
    active_units,
    encoder_r2,
    fit_free_energy,
    free_energy,
    residual_log_precision,
)
from .diagnostics import discriminability, map_statistics, statistic_gap
from .priors import (
    IIDGaussianPrior,
    NoPrior,
    SpatialGaussianPrior,
    SpectralGaussianPrior,
    StationaryGaussianPrior,
    SumPrior,
    VarianceFieldPrior,
)
from .wake_sleep import Recognition, fit_decoder, recon_r2, wake_sleep

__all__ = [
    "AmortizedEncoder",
    "TICAPrior",
    "active_units",
    "encoder_r2",
    "fit_free_energy",
    "free_energy",
    "ConvFactorAnalysis",
    "IIDGaussianPrior",
    "Recognition",
    "NoPrior",
    "SpatialGaussianPrior",
    "SpectralGaussianPrior",
    "SumPrior",
    "context_precision_fe",
    "expected_signal_precision",
    "residual_variance",
    "settle",
    "settle_lbfgs",
    "StationaryGaussianPrior",
    "VarianceFieldPrior",
    "discriminability",
    "fit_decoder",
    "map_statistics",
    "recon_r2",
    "residual_log_precision",
    "statistic_gap",
    "wake_sleep",
]
