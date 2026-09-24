"""Topographic ICA across channels (plan Phase 2, block step [6]).

Units live on a 2D sheet (a torus, so there are no edge effects). The loss
rewards sparse *local energy*: each unit's squared output is summed over its
sheet neighborhood, and the square root of that pooled energy is penalized.
Units whose energies co-vary (e.g. the two halves of a quadrature pair, or
neighboring orientations) end up adjacent on the sheet, like pinwheels in V1.

Two forms:
  * complete TICA  - whitened input, square orthonormal W (Hyvarinen & Hoyer 2001)
  * RICA           - unconstrained, possibly overcomplete W with a reconstruction
                     penalty in place of orthonormality (Le et al. 2011)

Gradient descent settles each unit on a feature early, after which moving
units around the sheet is a discrete swap the gradient cannot make, so maps
stay patchy. ``arrange_on_sheet`` fixes this with greedy pairwise swaps that
put units with correlated energies next to each other (this lowers the TICA
loss itself), followed by a short gradient polish.

    s      = W z                                  z: whitened input
    loss   = mean_samples sum_i sqrt(sum_j h_ij s_j**2 + eps)
    loss  += lam * mean ||W^T W z - z||**2        (RICA only)
"""

import math

import torch
from torch import nn


def torus_neighborhood(rows: int, cols: int, radius: int = 1) -> torch.Tensor:
    """(n, n) 0/1 matrix: units within a (2r+1)x(2r+1) square on a torus."""
    r = torch.arange(rows * cols) // cols
    c = torch.arange(rows * cols) % cols
    dr = (r[:, None] - r[None, :]).abs()
    dc = (c[:, None] - c[None, :]).abs()
    dr = torch.minimum(dr, rows - dr)
    dc = torch.minimum(dc, cols - dc)
    return ((dr <= radius) & (dc <= radius)).float()


def torus_distance(rows: int, cols: int) -> torch.Tensor:
    """(n, n) Chebyshev distance between units on the torus sheet."""
    r = torch.arange(rows * cols) // cols
    c = torch.arange(rows * cols) % cols
    dr = (r[:, None] - r[None, :]).abs()
    dc = (c[:, None] - c[None, :]).abs()
    return torch.maximum(torch.minimum(dr, rows - dr), torch.minimum(dc, cols - dc))


def tica_loss(s: torch.Tensor, h: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Mean over samples of sum_i sqrt(pooled local energy). s: (N, n)."""
    return torch.sqrt(s.pow(2) @ h.T + eps).sum(1).mean()


def symmetric_orthonormalize(w: torch.Tensor) -> torch.Tensor:
    """W <- (W W^T)^{-1/2} W, the nearest matrix with orthonormal rows."""
    u, _, vh = torch.linalg.svd(w, full_matrices=False)
    return u @ vh


class FittedModule(nn.Module):
    """Module whose buffers are sized by ``fit``; loading a checkpoint into a
    fresh (unfitted) instance resizes them to the saved shapes."""

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name, buf in self._buffers.items():
            key = prefix + name
            if buf is not None and key in state_dict and buf.shape != state_dict[key].shape:
                self._buffers[name] = torch.empty_like(state_dict[key])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Whitener(FittedModule):
    """Standardize each input dim, then PCA-whiten to ``dim`` components.

    ``transform`` maps (N, D) -> (N, dim); ``matrix``/``offset`` expose the same
    map as one affine transform z = x @ matrix.T + offset, so a stage can fold
    it into a convolution.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))
        self.register_buffer("components", torch.empty(0))  # (dim, D), rows = PCs
        self.register_buffer("variances", torch.empty(0))  # (dim,)

    def fit(self, x: torch.Tensor, dim: int) -> "Whitener":
        x = x.double()
        self.mean = x.mean(0)
        self.scale = x.std(0) + 1e-8
        xs = (x - self.mean) / self.scale
        cov = xs.T @ xs / (len(xs) - 1)
        evals, evecs = torch.linalg.eigh(cov)
        order = evals.argsort(descending=True)[:dim]
        self.variances = evals[order].float()
        self.components = evecs[:, order].T.float()
        self.mean, self.scale = self.mean.float(), self.scale.float()
        return self

    @property
    def matrix(self) -> torch.Tensor:
        return (self.components / self.variances.sqrt()[:, None]) / self.scale

    @property
    def offset(self) -> torch.Tensor:
        return -(self.matrix @ self.mean)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix.T + self.offset

    def explained_variance(self) -> float:
        """Fraction of the standardized variance kept (D = total variance)."""
        return float(self.variances.sum() / self.mean.numel())


class GroupWhitener(nn.Module):
    """Whiten channel groups with separate dimension budgets, then jointly
    re-whiten the concatenation at full rank.

    A single PCA over channel groups of very different size or variance lets
    the larger group crowd the smaller out of the kept subspace. Here each
    group ``(start, stop, dim)`` keeps its own ``dim`` components; the joint
    step only decorrelates across groups (it drops nothing), so TICA still sees
    white input. Same interface as ``Whitener``.
    """

    def __init__(self, groups: list[tuple[int, int, int]]):
        super().__init__()
        self.groups = [tuple(g) for g in groups]
        self.parts = nn.ModuleList(Whitener() for _ in groups)
        self.joint = Whitener()

    def fit(self, x: torch.Tensor, dim: int | None = None) -> "GroupWhitener":
        total = sum(d for _, _, d in self.groups)
        if dim is not None and dim != total:
            raise ValueError(f"group budgets sum to {total}, but dim={dim}")
        zs = [w.fit(x[:, a:b], d).transform(x[:, a:b]) for w, (a, b, d) in zip(self.parts, self.groups)]
        self.joint.fit(torch.cat(zs, 1), total)
        return self

    @property
    def matrix(self) -> torch.Tensor:
        n_in = max(b for _, b, _ in self.groups)
        rows = []
        for w, (a, b, _) in zip(self.parts, self.groups):
            m = torch.zeros(w.matrix.shape[0], n_in)
            m[:, a:b] = w.matrix
            rows.append(m)
        return self.joint.matrix @ torch.cat(rows, 0)

    @property
    def offset(self) -> torch.Tensor:
        inner = torch.cat([w.offset for w in self.parts])
        return self.joint.matrix @ inner + self.joint.offset

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix.T + self.offset

    def explained_variance(self) -> list[float]:
        """Per-group fraction of standardized variance kept."""
        return [w.explained_variance() for w in self.parts]


def fit_tica(
    z: torch.Tensor,
    h: torch.Tensor,
    n_iter: int = 500,
    lr: float = 0.5,
    eps: float = 1e-3,
    batch_size: int | None = None,
    seed: int = 0,
    init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Complete TICA on whitened ``z`` (N, d). Returns (W (d, d), loss history).

    Riemannian gradient descent on the orthogonal group: take the gradient's
    component tangent to the orthonormal-W manifold, step, re-orthonormalize.
    (Adam's per-element step scaling fights the projection and stalls.)
    """
    d = z.shape[1]
    if h.shape[0] != d:
        raise ValueError(f"complete TICA needs n_units == dim ({h.shape[0]} != {d})")
    gen = torch.Generator().manual_seed(seed)
    w = symmetric_orthonormalize(torch.randn(d, d, generator=gen) if init is None else init)
    hist = []
    for _ in range(n_iter):
        zb = z if batch_size is None else z[torch.randint(len(z), (batch_size,), generator=gen)]
        w.requires_grad_(True)
        loss = tica_loss(zb @ w.T, h, eps)
        (grad,) = torch.autograd.grad(loss, w)
        with torch.no_grad():
            w = symmetric_orthonormalize(w - lr * (grad - w @ grad.T @ w))
        hist.append(loss.item())
    return w, hist


def fit_rica(
    z: torch.Tensor,
    h: torch.Tensor,
    lam: float = 1.0,
    n_iter: int = 1000,
    lr: float = 0.01,
    eps: float = 1e-3,
    batch_size: int | None = None,
    seed: int = 0,
    init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Topographic RICA; ``n_units = h.shape[0]`` may exceed ``d``.
    Returns (W (n_units, d) with unit-norm rows, loss history).

    Rows are normalized inside the objective (as in the reference RICA code);
    otherwise the sparsity term is minimized by shrinking every unit toward
    zero. With unit rows the reconstruction term pushes W^T W toward a
    multiple of I, spreading overcomplete units evenly (a tight frame).
    """
    n, d = h.shape[0], z.shape[1]
    gen = torch.Generator().manual_seed(seed)
    w0 = torch.randn(n, d, generator=gen) if init is None else init.clone()
    w = w0.requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    hist = []
    for _ in range(n_iter):
        zb = z if batch_size is None else z[torch.randint(len(z), (batch_size,), generator=gen)]
        wn = w / w.norm(dim=1, keepdim=True)
        s = zb @ wn.T
        recon = (s @ wn - zb).pow(2).sum(1).mean()
        loss = tica_loss(s, h, eps) + lam * recon
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    w = w.detach()
    return w / w.norm(dim=1, keepdim=True), hist


def arrange_on_sheet(s: torch.Tensor, h: torch.Tensor, max_sweeps: int = 20) -> torch.Tensor:
    """Permutation ``perm`` (sheet position -> unit) that greedily maximizes
    sum_{neighbors} corr(s_i**2, s_j**2) by pairwise swaps. s: (N, n)."""
    e = torch.corrcoef(s.pow(2).T)
    e.fill_diagonal_(0)
    hh = h.clone()
    hh.fill_diagonal_(0)
    n = len(e)
    perm = torch.arange(n)
    for _ in range(max_sweeps):
        improved = False
        for a in range(n):
            for b in range(a + 1, n):
                # change in objective from swapping the units at positions a and b;
                # the a-b pair term itself is unchanged, so leave it out
                pa, pb = int(perm[a]), int(perm[b])
                diff = (hh[a] - hh[b]) * (e[pb][perm] - e[pa][perm])
                delta = 2 * (diff.sum() - diff[a] - diff[b])
                if delta > 1e-9:
                    perm[a], perm[b] = pb, pa
                    improved = True
        if not improved:
            break
    return perm


class TICA(FittedModule):
    """Fitted whitening + TICA/RICA weights on a ``rows`` x ``cols`` sheet."""

    def __init__(
        self,
        rows: int,
        cols: int,
        radius: int = 1,
        eps: float = 1e-3,
        whitener: nn.Module | None = None,
    ):
        super().__init__()
        self.rows, self.cols, self.eps = rows, cols, eps
        self.whitener = whitener if whitener is not None else Whitener()
        self.register_buffer("h", torus_neighborhood(rows, cols, radius))
        self.register_buffer("weight", torch.empty(0))

    @property
    def n_units(self) -> int:
        return self.rows * self.cols

    def fit(
        self,
        x: torch.Tensor,
        dim: int,
        mode: str = "complete",
        arrange: bool = True,
        polish_iter: int = 200,
        **kw,
    ) -> list[float]:
        """Fit on raw samples ``x`` (N, D). ``mode``: "complete" or "rica".
        With ``arrange``, units are re-seated on the sheet by energy correlation
        and then polished with ``polish_iter`` more gradient steps."""
        fitters = {"complete": fit_tica, "rica": fit_rica}
        if mode not in fitters:
            raise ValueError(f"unknown TICA mode {mode!r}")
        fit = fitters[mode]
        self.whitener.fit(x, dim)
        z = self.whitener.transform(x)
        w, hist = fit(z, self.h, eps=self.eps, **kw)
        if arrange:
            w = w[arrange_on_sheet(z @ w.T, self.h)]
            kw = {**kw, "n_iter": polish_iter, "init": w}
            w, more = fit(z, self.h, eps=self.eps, **kw)
            hist += more
        self.weight = w
        return hist

    @property
    def affine(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(matrix (n_units, D), bias (n_units,)) with s = x @ matrix.T + bias."""
        return self.weight @ self.whitener.matrix, self.weight @ self.whitener.offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, D) -> (N, n_units) TICA coefficients s."""
        return self.whitener.transform(x) @ self.weight.T

    def pooled_energy(self, s: torch.Tensor) -> torch.Tensor:
        """sqrt of neighborhood-pooled energy (the complex-cell-like output)."""
        return torch.sqrt(s.pow(2) @ self.h.T + self.eps)
