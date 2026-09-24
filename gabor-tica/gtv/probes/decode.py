"""Linear readouts for comparing representations (plan Phase 2 A-B comparison)."""

import torch


def _standardize(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return (train - mu) / sd, (test - mu) / sd


def train_test_split(n: int, train_frac: float, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(n, generator=gen)
    k = int(round(train_frac * n))
    return perm[:k], perm[k:]


def fit_logistic(
    x: torch.Tensor, y: torch.Tensor, n_classes: int, l2: float = 1e-3, n_iter: int = 300
) -> torch.nn.Linear:
    """Multinomial logistic regression fitted with L-BFGS."""
    model = torch.nn.Linear(x.shape[1], n_classes)
    opt = torch.optim.LBFGS(model.parameters(), max_iter=n_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y) + l2 * model.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return model


def linear_decode(
    x: torch.Tensor,
    y: torch.Tensor,
    train_frac: float = 0.6,
    l2: float = 1e-3,
    seed: int = 0,
) -> float:
    """Held-out accuracy of a linear classifier on features ``x`` (N, D)."""
    gen = torch.Generator().manual_seed(seed)
    tr, te = train_test_split(len(x), train_frac, gen)
    xtr, xte = _standardize(x[tr], x[te])
    n_classes = int(y.max()) + 1
    with torch.enable_grad():
        model = fit_logistic(xtr, y[tr], n_classes, l2)
    with torch.no_grad():
        return float((model(xte).argmax(1) == y[te]).float().mean())


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the ROC curve (probability a positive outranks a negative)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    ranks = torch.cat([pos, neg]).argsort().argsort().double() + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def linear_detect_auc(
    x: torch.Tensor, y: torch.Tensor, train_frac: float = 0.6, l2: float = 1e-3, seed: int = 0
) -> float:
    """Held-out AUC of a linear binary detector on features ``x`` (N, D)."""
    gen = torch.Generator().manual_seed(seed)
    tr, te = train_test_split(len(x), train_frac, gen)
    xtr, xte = _standardize(x[tr], x[te])
    with torch.enable_grad():
        model = fit_logistic(xtr, y[tr].long(), 2, l2)
    with torch.no_grad():
        s = model(xte)
        return auc(s[:, 1] - s[:, 0], y[te])
