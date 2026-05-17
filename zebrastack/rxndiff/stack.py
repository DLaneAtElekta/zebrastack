"""Hierarchical reaction-diffusion stack.

A stack of :class:`RDStage` instances where stage[k]'s product field is
used as a spatial ``feed_map`` for stage[k+1]. This realises the
"product of one reaction is consumed by the next" coupling: a region
that produces lots of V upstream provides extra substrate/source
downstream, so downstream patterning is spatially modulated by upstream
structure.
"""

import numpy as np


def normalize01(x):
    lo = x.min()
    hi = x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HierarchicalRDStack:
    """A cascade of RD stages with spatial-feed coupling."""

    def __init__(self, stages, normalize=True):
        self.stages = list(stages)
        if not self.stages:
            raise ValueError("Need at least one stage")
        sizes = {s.size for s in self.stages}
        if len(sizes) != 1:
            raise ValueError(f"All stages must share grid size; got {sizes}")
        self.size = sizes.pop()
        self.normalize = normalize

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        for s in self.stages:
            s.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))

    def step(self, n_steps=1, substeps_per_stage=1):
        """Advance the stack by ``n_steps`` outer iterations.

        ``substeps_per_stage`` may be a scalar (applied to every stage) or
        a list with one entry per stage — useful when stages have very
        different ``dt`` (e.g. Gray-Scott at dt=1 and FitzHugh-Nagumo at
        dt=1e-3 in the same cascade).
        """
        if np.isscalar(substeps_per_stage):
            sub = [int(substeps_per_stage)] * len(self.stages)
        else:
            sub = [int(s) for s in substeps_per_stage]
            if len(sub) != len(self.stages):
                raise ValueError(
                    f"substeps_per_stage length {len(sub)} != stages {len(self.stages)}")
        for _ in range(n_steps):
            for k, s in enumerate(self.stages):
                feed = None
                if k > 0:
                    prev = self.stages[k - 1].product
                    feed = normalize01(prev) if self.normalize else prev
                s.step(n_steps=sub[k], feed_map=feed)
        return self

    def fields(self):
        """Return ``[(U, V), ...]`` snapshots, one per stage."""
        return [(s.U.copy(), s.V.copy()) for s in self.stages]

    def outputs(self):
        """Return product fields for every stage (post-normalization)."""
        out = []
        for k, s in enumerate(self.stages):
            v = s.product
            out.append(normalize01(v) if self.normalize else v.copy())
        return out

    def output(self):
        """Product field of the final stage."""
        v = self.stages[-1].product
        return normalize01(v) if self.normalize else v.copy()
