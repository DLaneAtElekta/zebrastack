"""Drivers that search the parameter space of a hierarchical RD stack.

Two drivers are provided:

* :class:`PreferenceDriver` — render a small grid of variants from the
  current parameter centroid, let a human pick one, and perturb around
  the chosen variant. A simple breeder / interactive evolution loop.

* :class:`PerceptualDriver` — a (1, lambda) evolution strategy that
  maximises (or minimises) the Gabor-based perceptual distance between
  the stack's output and a target image.

Both drivers operate on parameter vectors built from the ``param_keys``
of each stage, so they work uniformly for Gray-Scott, FitzHugh-Nagumo,
or mixed stacks.
"""

import numpy as np

from .perceptual import perceptual_features


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


class PerceptualDriver:
    """(1, lambda) evolution strategy on perceptual distance.

    ``mode='maximize'`` searches for parameters whose rendered output is
    *far* in feature space from ``target`` (useful for exploring novel
    patterns relative to a baseline). ``mode='match'`` minimises the
    distance to the target.
    """

    def __init__(self, stack_factory, target=None, mode='maximize',
                 n_steps=2000, sigma=0.08, n_offspring=8, render_seed=0,
                 bank=None):
        if mode not in ('maximize', 'match'):
            raise ValueError("mode must be 'maximize' or 'match'")
        self.stack_factory = stack_factory
        self.mode = mode
        self.n_steps = n_steps
        self.sigma = sigma
        self.n_offspring = n_offspring
        self.render_seed = render_seed
        self.bank = bank

        base = stack_factory()
        self.params = get_stack_params(base)
        self.layout = stack_param_layout(base)

        if target is None:
            target = self._render(self.params)
        self.target = target
        self.target_feat = perceptual_features(target, bank=self.bank)
        self.best_image = self._render(self.params)
        self.best_distance = float(np.linalg.norm(
            perceptual_features(self.best_image, bank=self.bank)
            - self.target_feat))
        self.history = [self.best_distance]

    def _render(self, params):
        stack = self.stack_factory()
        set_stack_params(stack, params)
        stack.reset(seed=self.render_seed)
        stack.step(n_steps=self.n_steps)
        return stack.output()

    def _distance(self, image):
        f = perceptual_features(image, bank=self.bank)
        return float(np.linalg.norm(f - self.target_feat))

    def step(self, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        best = None  # (score, child_params, image, distance)
        for _ in range(self.n_offspring):
            child = self.params * (1.0 + rng.normal(0.0, self.sigma,
                                                    size=self.params.shape))
            img = self._render(child)
            d = self._distance(img)
            score = d if self.mode == 'maximize' else -d
            if best is None or score > best[0]:
                best = (score, child, img, d)
        improved = (best[3] > self.best_distance) if self.mode == 'maximize' \
            else (best[3] < self.best_distance)
        if improved:
            self.params = best[1]
            self.best_image = best[2]
            self.best_distance = best[3]
        self.history.append(self.best_distance)
        return best[2], best[3]

    def run(self, n_generations=20, rng=None, verbose=False):
        rng = np.random.default_rng() if rng is None else rng
        for g in range(n_generations):
            img, d = self.step(rng=rng)
            if verbose:
                print(f"gen {g:3d}  dist={d:.4f}  best={self.best_distance:.4f}")
        return self.best_image, self.best_distance
