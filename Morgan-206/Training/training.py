"""Train one fold and predict the held-out rows.

The network is one hidden layer of five tanh units, inputs and targets scaled
to [-1, 1], an ensemble of ten members each taken as the best of five random
restarts.

The gradient-boosted tree learner reported alongside the network in the
manuscript lives here too, so a fold is one call either way.

Because the network's initialisation is random, retraining reproduces the
published predictions only to within run-to-run variation; the stored
``YPRED_*.csv`` vectors are what reproduce them exactly. XGBoost enables no row
or column subsampling, so it is a deterministic function of the data and the
seed and does return its predictions bit for bit.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- the network

HIDDEN = 5
MAX_EPOCHS = 400
PATIENCE = 10      # in validation checks, so PATIENCE * VAL_EVERY epochs
VAL_EVERY = 5      # a full validation pass costs about one epoch of training
LEARNING_RATE = 3e-4
BATCH = 512

N_MEMBERS = 10
N_RESTARTS = 5

#: base seed; each fold is offset so the five folds do not share an
#: initialisation, matching the published run
ANN_SEED = 20_260_728
FOLD_STRIDE = 17

#: XGBoost draws no randomness here, so every fold shares one seed
XGB_SEED = 20_260_724

XGB_PARAMS = {
    "n_estimators": 2_500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 1,
}


def fold_seed(algorithm: str, fold: int) -> int:
    return ANN_SEED + FOLD_STRIDE * fold if algorithm == "ANN" else XGB_SEED


def _minmax_fit(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Offset and span that map each column onto [-1, 1].

    A constant column has zero span; dividing by one instead sends it to a
    constant, which the network can absorb into the bias.
    """
    lo, hi = a.min(0), a.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return lo.astype(np.float32), span.astype(np.float32)


def train_fold_ann(x_train: np.ndarray, y_train: np.ndarray,
                   x_test: np.ndarray, seed: int,
                   n_members: int = N_MEMBERS, n_restarts: int = N_RESTARTS,
                   device: str = "cuda", **_) -> np.ndarray:
    """Ensemble prediction for ``x_test``, in the original target units.

    Each candidate divides its training rows 80 / 10 / 10 into a fit split, a
    validation split that drives early stopping, and an internal test split
    that decides which restart represents each ensemble member. All fifty
    candidates of a fold are trained together as one batched tensor, so a fold
    is a single GPU job rather than fifty.
    """
    import torch

    k = n_members * n_restarts
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=dev).manual_seed(int(seed))

    x_lo, x_span = _minmax_fit(x_train)
    y_lo = float(y_train.min())
    y_span = float(y_train.max() - y_train.min()) or 1.0

    xtr = torch.as_tensor((x_train - x_lo) / x_span * 2.0 - 1.0,
                          dtype=torch.float32, device=dev)
    xte = torch.as_tensor((x_test - x_lo) / x_span * 2.0 - 1.0,
                          dtype=torch.float32, device=dev)
    ytr = torch.as_tensor((y_train - y_lo) / y_span * 2.0 - 1.0,
                          dtype=torch.float32, device=dev)

    n, p = xtr.shape

    # each candidate gets its own 80/10/10 division of the training rows
    order = torch.argsort(torch.rand(k, n, generator=g, device=dev), dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(n, device=dev).expand(k, n))
    n_fit = int(round(0.8 * n))
    n_val = max(1, int(round(0.1 * n)))
    m_fit = (rank < n_fit).float()
    m_val = ((rank >= n_fit) & (rank < n_fit + n_val)).float()
    m_test = (rank >= n_fit + n_val).float()

    # Glorot bound: keeps tanh out of saturation for p ~ 1.5k binary inputs
    bound = float(np.sqrt(6.0 / (p + HIDDEN)))
    w1 = ((torch.rand(k, p, HIDDEN, generator=g, device=dev) * 2 - 1)
          * bound).requires_grad_(True)
    b1 = torch.zeros(k, 1, HIDDEN, device=dev, requires_grad=True)
    w2 = (torch.rand(k, HIDDEN, 1, generator=g, device=dev) * 2 - 1
          ).requires_grad_(True)
    b2 = torch.zeros(k, 1, 1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w1, b1, w2, b2], lr=LEARNING_RATE)
    count_val = m_val.sum(1).clamp(min=1)

    def forward(xb, a1, c1, a2, c2):
        """Forward for a row block shared by all k candidates -> [k, m].

        All candidates see the same rows, so the first layer is one dense
        [m, p] x [p, k*HIDDEN] matmul. Broadcasting to [k, m, p] instead would
        materialise about 150 MB per step and dominate the runtime.
        """
        m = xb.shape[0]
        h = xb @ a1.permute(1, 0, 2).reshape(p, k * HIDDEN)
        h = torch.tanh(h.view(m, k, HIDDEN).permute(1, 0, 2) + c1)
        return torch.baddbmm(c2, h, a2).squeeze(-1)

    best_val = torch.full((k,), float("inf"), device=dev)
    best = [t.detach().clone() for t in (w1, b1, w2, b2)]
    stale = 0
    steps = max(1, n // BATCH)

    for epoch in range(MAX_EPOCHS):
        perm = torch.randperm(n, generator=g, device=dev)
        # one gather per epoch; the per-step slices below are contiguous views
        xs, ys, ms = xtr[perm], ytr[perm], m_fit[:, perm]
        for s in range(steps):
            lo_i, hi_i = s * BATCH, min((s + 1) * BATCH, n)
            if hi_i <= lo_i:
                continue
            opt.zero_grad(set_to_none=True)
            out = forward(xs[lo_i:hi_i], w1, b1, w2, b2)
            sq = (out - ys[lo_i:hi_i].view(1, -1)) ** 2
            wt = ms[:, lo_i:hi_i]
            (((sq * wt).sum(1) / wt.sum(1).clamp(min=1)).sum()).backward()
            opt.step()

        if epoch % VAL_EVERY and epoch != MAX_EPOCHS - 1:
            continue
        with torch.no_grad():
            sq = (forward(xtr, w1, b1, w2, b2) - ytr.view(1, -1)) ** 2
            val = (sq * m_val).sum(1) / count_val
            val = torch.nan_to_num(val, nan=float("inf"), posinf=float("inf"))
            better = val < best_val
            if bool(better.any()):
                best_val = torch.where(better, val, best_val)
                sel = better.view(k, 1, 1)
                for store, live in zip(best, (w1, b1, w2, b2)):
                    store.copy_(torch.where(sel, live.detach(), store))
                stale = 0
            else:
                stale += 1
        if stale >= PATIENCE:
            break

    with torch.no_grad():
        bw1, bb1, bw2, bb2 = best
        # the internal-test split decides which restart represents each member
        sq = (forward(xtr, bw1, bb1, bw2, bb2) - ytr.view(1, -1)) ** 2
        test_mse = (sq * m_test).sum(1) / m_test.sum(1).clamp(min=1)
        out_te = forward(xte, bw1, bb1, bw2, bb2)
        pick = test_mse.view(n_members, n_restarts).argmin(dim=1)
        idx = torch.arange(n_members, device=dev) * n_restarts + pick
        scaled = out_te[idx].mean(0)

    return ((scaled.cpu().numpy() + 1.0) / 2.0 * y_span + y_lo).astype(np.float64)


# ------------------------------------------------------------ the tree model

def train_fold_xgboost(x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, seed: int = XGB_SEED,
                       n_jobs: int = 4, **_) -> np.ndarray:
    """Fit on the training rows and predict the held-out fold.

    XGBoost was selected as the representative tree-based learner after
    comparing it against LightGBM, random forests and RBF-SVR on identical
    inputs and partitions. The settings are stated here rather than tuned at
    run time, so a rerun depends on nothing outside this file.
    """
    from scipy.sparse import csr_matrix
    from xgboost import XGBRegressor

    model = XGBRegressor(
        **XGB_PARAMS,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=int(seed),
        verbosity=0,
    )
    model.fit(csr_matrix(x_train), y_train)
    return np.asarray(model.predict(csr_matrix(x_test)), dtype=np.float64).ravel()


LEARNERS = {"ANN": train_fold_ann, "XGBoost": train_fold_xgboost}


def training(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
             algorithm: str, fold: int, **kwargs) -> np.ndarray:
    """Train one fold with the named learner and predict the held-out rows."""
    try:
        learner = LEARNERS[algorithm]
    except KeyError:
        raise ValueError(f"unknown algorithm {algorithm!r}; "
                         f"choose from {', '.join(LEARNERS)}") from None
    return learner(x_train, y_train, x_test,
                   seed=fold_seed(algorithm, fold), **kwargs)
