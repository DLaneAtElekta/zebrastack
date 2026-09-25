"""Top-down generative path through the V4 -> PIT -> AIT stack (Phase 5
deferral, needed for Phase 6 templates).

One convolutional factor-analysis decoder per step (upper stage outputs ->
lower stage outputs, bilinear 2x upsampling + conv), fitted by regression on
the stack's own recognized activity (the Phase 3a approach). Templates:
clamp AIT to a category's mean activity, decode down, and read the lower
stages' implied activity.
"""

import torch

from .decoders import ConvFactorAnalysis
from .wake_sleep import fit_decoder


def fit_topdown(maps: dict[str, torch.Tensor], order: list[str], kernel: int = 3, n_iter: int = 400,
                lr: float = 0.02, generator=None) -> dict[str, ConvFactorAnalysis]:
    """``maps``: stage name -> recognized outputs (N, C, H, W) on the same
    images; ``order``: bottom to top, e.g. ["V4", "PIT", "AIT"]. Returns
    decoders keyed "upper->lower"."""
    decoders = {}
    for lower, upper in zip(order[:-1], order[1:]):
        x, z = maps[lower], maps[upper]
        dec = ConvFactorAnalysis(z.shape[1], x.shape[1], kernel)
        dec.fit_standardizer(x)
        fit_decoder(dec, dec.from_v1(x), z, n_iter=n_iter, lr=lr, border=0, generator=generator)
        decoders[f"{upper}->{lower}"] = dec
    return decoders


def decode_down(top: torch.Tensor, decoders: dict, order: list[str]) -> dict[str, torch.Tensor]:
    """Run clamped top-level activity down the stack: returns expected outputs
    for every stage in ``order`` (the top included)."""
    out = {order[-1]: top}
    cur = top
    for lower, upper in zip(reversed(order[:-1]), reversed(order[1:])):
        dec = decoders[f"{upper}->{lower}"]
        with torch.no_grad():
            cur = dec.to_v1(dec.mean(cur))
        out[lower] = cur
    return out
