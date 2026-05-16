"""Reaction-diffusion stages.

Each stage maintains two scalar fields (U, V) on a 2D grid and advances them
with explicit Euler integration. Subclasses implement ``_kinetics(U, V,
feed_map)`` and declare ``param_keys`` so drivers can read/write parameters
as a flat vector.
"""

import numpy as np


def laplacian(Z, dx=1.0):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4.0 * Zcenter) / (dx * dx)


class RDStage:
    """Base class for a 2D two-field reaction-diffusion stage."""

    param_keys = ()

    def __init__(self, size=96, dx=1.0, dt=1.0, seed=None):
        self.size = size
        self.dx = dx
        self.dt = dt
        self.U = np.empty((size, size), dtype=np.float64)
        self.V = np.empty((size, size), dtype=np.float64)
        self.reset(seed=seed)

    def reset(self, seed=None):
        raise NotImplementedError

    def _kinetics(self, U, V, feed_map):
        raise NotImplementedError

    def step(self, n_steps=1, feed_map=None):
        for _ in range(n_steps):
            lapU = laplacian(self.U, self.dx)
            lapV = laplacian(self.V, self.dx)
            Uc = self.U[1:-1, 1:-1]
            Vc = self.V[1:-1, 1:-1]
            fm = None if feed_map is None else feed_map[1:-1, 1:-1]
            dU, dV = self._kinetics(Uc, Vc, fm)
            self.U[1:-1, 1:-1] = Uc + self.dt * (self.Du * lapU + dU)
            self.V[1:-1, 1:-1] = Vc + self.dt * (self.Dv * lapV + dV)
            # Neumann boundary
            self.U[0, :] = self.U[1, :]
            self.U[-1, :] = self.U[-2, :]
            self.U[:, 0] = self.U[:, 1]
            self.U[:, -1] = self.U[:, -2]
            self.V[0, :] = self.V[1, :]
            self.V[-1, :] = self.V[-2, :]
            self.V[:, 0] = self.V[:, 1]
            self.V[:, -1] = self.V[:, -2]
        return self

    @property
    def product(self):
        """Field that is the 'output' of this stage; default is V."""
        return self.V


class GrayScottStage(RDStage):
    """Gray-Scott model.

        dU/dt = Du * lap(U) - U V^2 + F (1 - U)
        dV/dt = Dv * lap(V) + U V^2 - (F + k) V

    When a ``feed_map`` is supplied (typically the product of the prior
    stage, normalized to [0, 1]) the effective feed rate becomes
    ``F + F_alpha * feed_map``, so a high product upstream drives more
    substrate replenishment downstream.
    """

    param_keys = ("F", "k", "Du", "Dv", "F_alpha")

    def __init__(self, size=96, F=0.0367, k=0.0649, Du=0.16, Dv=0.08,
                 F_alpha=0.04, dx=1.0, dt=1.0, seed=None):
        self.F = F
        self.k = k
        self.Du = Du
        self.Dv = Dv
        self.F_alpha = F_alpha
        super().__init__(size=size, dx=dx, dt=dt, seed=seed)

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.U[:] = 1.0
        self.V[:] = 0.0
        n_blobs = int(rng.integers(8, 24))
        for _ in range(n_blobs):
            r0 = int(rng.integers(4, self.size - 4))
            c0 = int(rng.integers(4, self.size - 4))
            self.U[r0 - 3:r0 + 3, c0 - 3:c0 + 3] = 0.5
            self.V[r0 - 3:r0 + 3, c0 - 3:c0 + 3] = 0.25
        # tiny perturbation everywhere
        self.U += 0.01 * rng.standard_normal(self.U.shape)
        self.V += 0.01 * rng.standard_normal(self.V.shape)

    def _kinetics(self, U, V, feed_map):
        F_eff = self.F if feed_map is None else self.F + self.F_alpha * feed_map
        UV2 = U * V * V
        dU = -UV2 + F_eff * (1.0 - U)
        dV = UV2 - (F_eff + self.k) * V
        return dU, dV


class FitzHughNagumoStage(RDStage):
    """FitzHugh-Nagumo style RD (matches examples/rxndiff.ipynb).

        dU/dt = a * lap(U) + U - U^3 - V + k
        dV/dt = (b * lap(V) + U - V) / tau

    The effective ``k`` is offset by ``k_alpha * feed_map`` so that a strong
    upstream product locally biases the resting state.
    """

    param_keys = ("a", "b", "tau", "k", "k_alpha")

    def __init__(self, size=96, a=1.54e-4, b=4.7e-3, tau=1.946, k=-2.85e-3,
                 k_alpha=5e-3, dx=2.0 / 96, dt=1e-3, seed=None):
        # Defaults are the pre-scaled values from examples/rxndiff.ipynb,
        # which give well-formed labyrinth patterns under explicit Euler.
        self.a = a
        self.b = b
        self.tau = tau
        self.k = k
        self.k_alpha = k_alpha
        # diffusion coefficients used by the base step()
        self.Du = a
        self.Dv = b / tau
        super().__init__(size=size, dx=dx, dt=dt, seed=seed)

    def _refresh_diffusion(self):
        self.Du = self.a
        self.Dv = self.b / max(self.tau, 1e-8)

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.U[:] = rng.random((self.size, self.size))
        self.V[:] = rng.random((self.size, self.size))

    def step(self, n_steps=1, feed_map=None):
        # Keep Du/Dv consistent with current a/b/tau (drivers may mutate them)
        self._refresh_diffusion()
        return super().step(n_steps=n_steps, feed_map=feed_map)

    def _kinetics(self, U, V, feed_map):
        k_eff = self.k if feed_map is None else self.k + self.k_alpha * feed_map
        dU = U - U ** 3 - V + k_eff
        dV = (U - V) / max(self.tau, 1e-8)
        return dU, dV
