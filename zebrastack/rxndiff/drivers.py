"""Drivers that search the parameter space of a hierarchical RD stack.

Two drivers are provided:

* :class:`PreferenceDriver` — render a small grid of variants from the
  current parameter centroid, let a human pick one, and perturb around
  the chosen variant. A simple breeder / interactive evolution loop.

* :class:`PerceptualDriver` — a (1, lambda) evolution strategy with
  several scoring modes:

    - ``'maximize'``: push the rendered output far (in Gabor feature
      space) from a baseline / target.
    - ``'match'``: minimise distance to a target.
    - ``'structure'``: maximise a multi-scale Gabor *richness* score
      (see :func:`structure_score`) — rewards multi-scale, multi-orientation
      textures and explicitly rejects near-uniform / degenerate outputs.
    - ``'novelty'``: archive-based novelty search; each generation is
      scored by mean distance to its k nearest neighbours in an archive
      of accepted designs, so the search is pushed toward genuinely new
      patterns instead of drifting around a single anchor.
    - ``'novelty+structure'``: weighted sum of novelty and structure.
      Best default for visually-interesting exploration.

Both drivers operate on parameter vectors built from the ``param_keys``
of each stage, so they work uniformly for Gray-Scott, FitzHugh-Nagumo,
or mixed stacks.
"""

import numpy as np

from .perceptual import perceptual_features, structure_score


def get_stage_params(stage):
    return np.array([getattr(stage, k) for k in stage.param_keys],
                    dtype=np.float64)


def set_stage_params(stage, vec):
    for k, v in zip(stage.param_keys, vec):
        setattr(stage, k, float(v))


def get_stack_params(stack):
    return np.concatenate([get_stage_params(s) for s in stack.stages])


def set_stack_params(stack, vec):
    i = 0
    for s in stack.stages:
        n = len(s.param_keys)
        set_stage_params(s, vec[i:i + n])
        i += n


def stack_param_layout(stack):
    """Return [(stage_idx, key), ...] in the order params are flattened."""
    layout = []
    for i, s in enumerate(stack.stages):
        for k in s.param_keys:
            layout.append((i, k))
    return layout


class PreferenceDriver:
    """Interactive breeder: render N variants, user picks one, repeat."""

    def __init__(self, stack_factory, n_variants=9, perturb=0.10,
                 n_steps=2000, render_seed=0):
        self.stack_factory = stack_factory
        self.n_variants = n_variants
        self.perturb = perturb
        self.n_steps = n_steps
        self.render_seed = render_seed
        base = stack_factory()
        self.params = get_stack_params(base)
        self.layout = stack_param_layout(base)
        self.history = [self.params.copy()]

    def _render(self, params):
        stack = self.stack_factory()
        set_stack_params(stack, params)
        stack.reset(seed=self.render_seed)
        stack.step(n_steps=self.n_steps)
        return stack.output()

    def generate(self, rng=None):
        """Return (children_params, rendered_images). The first child is
        always the current centroid so the user can compare against it.
        """
        rng = np.random.default_rng() if rng is None else rng
        children = [self.params.copy()]
        for _ in range(self.n_variants - 1):
            jitter = 1.0 + rng.normal(0.0, self.perturb, size=self.params.shape)
            children.append(self.params * jitter)
        images = [self._render(p) for p in children]
        return children, images

    def pick(self, children, choice_idx):
        """Advance the centroid to the chosen child."""
        self.params = np.asarray(children[choice_idx], dtype=np.float64).copy()
        self.history.append(self.params.copy())
        return self.params


_MODES = ('maximize', 'match', 'structure', 'novelty', 'novelty+structure')


class PerceptualDriver:
    """(1, lambda) evolution strategy with multiple scoring modes.

    Parameters
    ----------
    stack_factory : callable
        Returns a fresh :class:`HierarchicalRDStack`.
    mode : str
        One of ``'maximize'``, ``'match'``, ``'structure'``,
        ``'novelty'``, ``'novelty+structure'``.
    target : ndarray, optional
        Target image for ``'maximize'``/``'match'`` modes. If omitted
        for ``'maximize'``, the initial rendered image is used as the
        baseline to flee from.
    n_steps : int
        Outer steps per render.
    sigma : float
        Multiplicative jitter standard deviation for offspring.
    n_offspring : int
        Population size per generation.
    archive_k : int
        For novelty modes, the number of nearest neighbours to average
        when scoring novelty.
    archive_threshold : float
        Minimum nearest-neighbour distance for a new design to be added
        to the archive. ``0.0`` adds every accepted design.
    structure_weight : float
        Weight of structure_score relative to novelty in
        ``'novelty+structure'``.
    structure_kw : dict, optional
        Extra keyword args passed through to :func:`structure_score`
        (e.g. ``min_std``, ``freqs``).
    bank : list, optional
        Pre-built Gabor bank for ``perceptual_features``.
    """

    def __init__(self, stack_factory, mode='structure', target=None,
                 n_steps=2000, sigma=0.08, n_offspring=8, render_seed=0,
                 bank=None, archive_k=3, archive_threshold=0.0,
                 structure_weight=0.5, structure_kw=None):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}; got {mode!r}")
        self.stack_factory = stack_factory
        self.mode = mode
        self.n_steps = n_steps
        self.sigma = sigma
        self.n_offspring = n_offspring
        self.render_seed = render_seed
        self.bank = bank
        self.archive_k = archive_k
        self.archive_threshold = archive_threshold
        self.structure_weight = structure_weight
        self.structure_kw = dict(structure_kw or {})

        base = stack_factory()
        self.params = get_stack_params(base)
        self.layout = stack_param_layout(base)

        if mode in ('maximize', 'match'):
            if target is None:
                target = self._render(self.params)
            self.target = target
            self.target_feat = perceptual_features(target, bank=self.bank)
        else:
            self.target = None
            self.target_feat = None

        self._archive_features = []
        self._archive_images = []

        self.best_image = self._render(self.params)
        self.best_score = self._score(self.best_image)
        if 'novelty' in mode:
            self._archive_features.append(
                perceptual_features(self.best_image, bank=self.bank))
            self._archive_images.append(self.best_image.copy())
        self.history = [self.best_score]

    # ----- rendering -----

    def _render(self, params):
        stack = self.stack_factory()
        set_stack_params(stack, params)
        stack.reset(seed=self.render_seed)
        stack.step(n_steps=self.n_steps)
        return stack.output()

    # ----- scoring -----

    def _novelty(self, feat):
        if not self._archive_features:
            return 1.0
        dists = np.array([np.linalg.norm(feat - af)
                          for af in self._archive_features])
        k = min(self.archive_k, dists.size)
        return float(np.mean(np.sort(dists)[:k]))

    def _score(self, image):
        if not np.isfinite(image).all():
            return -np.inf
        if self.mode == 'maximize':
            f = perceptual_features(image, bank=self.bank)
            return float(np.linalg.norm(f - self.target_feat))
        if self.mode == 'match':
            f = perceptual_features(image, bank=self.bank)
            return -float(np.linalg.norm(f - self.target_feat))
        if self.mode == 'structure':
            return structure_score(image, **self.structure_kw)
        if self.mode == 'novelty':
            f = perceptual_features(image, bank=self.bank)
            return self._novelty(f)
        if self.mode == 'novelty+structure':
            f = perceptual_features(image, bank=self.bank)
            n = self._novelty(f)
            s = structure_score(image, **self.structure_kw)
            return n + self.structure_weight * s
        raise RuntimeError(self.mode)

    # ----- main loop -----

    def step(self, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        candidates = []
        for _ in range(self.n_offspring):
            child = self.params * (1.0 + rng.normal(0.0, self.sigma,
                                                    size=self.params.shape))
            img = self._render(child)
            sc = self._score(img)
            candidates.append((sc, child, img))
        best = max(candidates, key=lambda c: c[0])
        best_score, best_params, best_image = best

        if 'novelty' in self.mode:
            # Always move centroid to the most-novel offspring; the score
            # surface shifts as the archive grows, so 'not strictly improved'
            # is the norm.
            self.params = best_params
            self.best_image = best_image
            self.best_score = best_score
            f = perceptual_features(best_image, bank=self.bank)
            if not self._archive_features:
                self._archive_features.append(f)
                self._archive_images.append(best_image.copy())
            else:
                nearest = min(np.linalg.norm(f - af)
                              for af in self._archive_features)
                if nearest > self.archive_threshold:
                    self._archive_features.append(f)
                    self._archive_images.append(best_image.copy())
        else:
            # Hill-climb: only accept strict improvements.
            if best_score > self.best_score:
                self.params = best_params
                self.best_image = best_image
                self.best_score = best_score

        self.history.append(self.best_score)
        return best_image, best_score

    def run(self, n_generations=20, rng=None, verbose=False):
        rng = np.random.default_rng() if rng is None else rng
        for g in range(n_generations):
            img, sc = self.step(rng=rng)
            if verbose:
                print(f"gen {g:3d}  score={sc:.4f}  "
                      f"best={self.best_score:.4f}  "
                      f"archive={len(self._archive_features)}")
        return self.best_image, self.best_score

    # ----- archive access -----

    @property
    def archive_images(self):
        return list(self._archive_images)

    @property
    def archive_size(self):
        return len(self._archive_images)
