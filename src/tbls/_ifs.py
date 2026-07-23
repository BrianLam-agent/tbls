"""Intuitionistic Fuzzy Set (IFS) score computation.

Two distinct IFS formulations coexist in the literature this package follows
and are kept here side by side:

- :func:`compute_if_scores_geib`: the GEIB formulation (Chen et al., IEEE TFS
  2025) used by :class:`tbls.tbls.TBLS`. Returns a diagonal weight matrix ``S``.
- :func:`compute_if_scores_simple`: the simplified
  membership/non-membership/hesitancy formulation used by
  :class:`tbls.gfcca.GraphFuzzyKCCA`. Returns a weight vector ``s``.

Cython acceleration candidate: the per-sample neighborhood loop computing
``Lambda[i] = mean(y[neighbors] != y[i])`` is Python-level O(n * k).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]

from . import _kernel


def compute_if_scores_geib(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    K: NDArray[np.float64] | None = None,
    if_sigma: float = 1.0,
) -> NDArray[np.float64]:
    """GEIB IFS scores as a diagonal weight matrix.

    Implements the formulas from Chen et al., IEEE TFS 2025 (GEIB). The returned
    diagonal matrix ``S`` holds per-sample scores in ``[eps, 1]`` used to weight
    the ridge regression in :class:`tbls.tbls.TBLS`.

    Args:
        X: Sample matrix of shape ``(n, d)``.
        y: Integer class labels of shape ``(n,)``.
        K: Precomputed kernel matrix of shape ``(n, n)``. If ``None``, it is
            computed from ``X`` via :func:`tbls._kernel.compute_kernel_matrix`.
        if_sigma: Scaling factor for the neighborhood radius.

    Returns:
        Diagonal weight matrix ``S`` of shape ``(n, n)``.
    """
    n = X.shape[0]
    if K is None:
        K = _kernel.compute_kernel_matrix(X)

    classes = np.unique(y)
    class_idx = {c: np.where(y == c)[0] for c in classes}
    class_cnt = {c: len(idx) for c, idx in class_idx.items()}

    # Per class: sum_{j in c} K(i,j) and mean_{j,k in c} K(j,k)
    class_sum_k = {c: K[:, idx].sum(axis=1) for c, idx in class_idx.items()}
    class_mean_k: dict[np.int64, float] = {}
    for c, idx in class_idx.items():
        if len(idx) > 0:
            k_cc = K[np.ix_(idx, idx)]
            class_mean_k[c] = float(k_cc.mean())
        else:
            class_mean_k[c] = 0.0

    # 1. Membership mu
    mu = np.zeros(n, dtype=np.float64)
    epsilon = 1e-8
    for c in classes:
        idx_c = class_idx[c]
        nc = class_cnt[c]
        # Distance to class center in kernel space
        dist_sq = np.diag(K)[idx_c] - 2.0 / nc * class_sum_k[c][idx_c] + class_mean_k[c]
        dist_sq = np.maximum(dist_sq, 0)
        dist = np.sqrt(dist_sq)
        r_c = dist.max() if len(dist) > 0 else 0.0
        # Numerical guard: if R_c is zero, all points sit at the center -> mu = 1
        if r_c < epsilon:
            mu[idx_c] = 1.0
        else:
            mu[idx_c] = 1.0 - dist / (r_c + epsilon)
    mu = np.clip(mu, 0.0, 1.0)

    # 2. Non-membership nu
    kernel_dists = _kernel.kernel_distance_matrix(K)
    off_diag = kernel_dists[~np.eye(n, dtype=bool)]
    median_dist = np.median(off_diag) if len(off_diag) > 0 else 1.0
    sigma = if_sigma * median_dist

    # Vectorized neighbor-mismatch rate: lambda_[i] = mean(y[neighbors] != y[i])
    # over neighbors within `sigma` (excluding self). `np.divide(..., where=...)`
    # reproduces the loop's "0.0 if no neighbors" branch without a 0/0 warning.
    neighbor_mask = (kernel_dists <= sigma) & ~np.eye(n, dtype=bool)
    diff = y[:, None] != y[None, :]
    neighbor_counts = neighbor_mask.sum(axis=1)
    mismatch_counts = (neighbor_mask & diff).sum(axis=1)
    lambda_ = np.divide(
        mismatch_counts,
        neighbor_counts,
        out=np.zeros(n, dtype=np.float64),
        where=neighbor_counts > 0,
    )
    nu = (1.0 - mu) * lambda_
    # Enforce mu + nu <= 1
    violation = mu + nu > 1.0
    if np.any(violation):
        nu[violation] = 1.0 - mu[violation] - 1e-8
    nu = np.clip(nu, 0.0, 1.0)

    # 3. IFS score
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if nu[i] < 1e-12:
            scores[i] = mu[i]
        elif mu[i] <= nu[i]:
            scores[i] = 0.0
        else:
            scores[i] = (1.0 - nu[i]) / (2.0 - mu[i] - nu[i])
    scores = np.clip(scores, 1e-6, 1.0)
    return np.diag(scores)


def compute_if_scores_simple(
    A: NDArray[np.float64],
    y: NDArray[np.int64],
    sigma_if: float = 1.0,
    delta_if: float = 0.5,
    min_weight: float = 1e-4,
) -> NDArray[np.float64]:
    """Simplified IFS scores as a weight vector.

    Used by :class:`tbls.gfcca.GraphFuzzyKCCA`. Both ``sigma_if`` and
    ``delta_if`` are *relative* to the data's median pairwise Euclidean distance
    (multiplied by it) so neither depends on the absolute feature scale or
    dimensionality -- ``sigma_if`` is the Gaussian width for the membership
    term, ``delta_if`` is the neighborhood threshold.

    Args:
        A: Sample matrix of shape ``(n, d)``.
        y: Integer class labels of shape ``(n,)``.
        sigma_if: Relative Gaussian width for the membership computation.
            Interpreted as ``sigma_if * median_pairwise_distance`` (an absolute
            distance unit here would numerically underflow ``mu`` to zero on
            any real, non-toy-scale dataset). Matches ``delta_if``'s relativity.
        delta_if: Relative distance threshold for the neighborhood.
        min_weight: Minimum clipping value; prevents zero weights that would
            make the regularized matrix singular.

    Returns:
        Weight vector ``s`` of shape ``(n,)``, clipped to ``[min_weight, 1]``.
    """
    n = A.shape[0]
    classes = np.unique(y)
    centers: dict[np.int64, NDArray[np.float64]] = {}
    for c in classes:
        idx_c = y == c
        centers[c] = A[idx_c].mean(axis=0)

    # Relative distance scale (shared by both the membership term and the
    # neighborhood threshold below) -- computed once, up front.
    dists = cdist(A, A, "euclidean")
    off_diag = dists[~np.eye(n, dtype=bool)]
    median_dist = np.median(off_diag) if off_diag.size > 0 else 1.0

    # Membership: Gaussian in units of the data's own median pairwise distance,
    # matching ``delta_if`` below (and ``compute_if_scores_geib``'s
    # ``sigma = if_sigma * median_dist``) -- NOT an absolute distance unit (an
    # absolute ``sigma_if`` underflows ``mu`` to numerical zero on any real,
    # non-toy-scale dataset, which collapses the IFS weights to ``min_weight``).
    sigma_eff = sigma_if * median_dist
    mu = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ci = y[i]
        dist = np.linalg.norm(A[i] - centers[ci])
        mu[i] = np.exp(-(dist**2) / (2 * sigma_eff**2))
    mu = np.clip(mu, 0.0, 1.0)

    threshold = median_dist * delta_if

    # Vectorized neighbor-mismatch rate: rho[i] = mean(y[neighbors] != y[i])
    # over neighbors strictly closer than `threshold` (excluding self). Strict
    # inequality matches the loop; `np.divide(..., where=...)` reproduces the
    # "0.0 if no neighbors" branch without a 0/0 warning.
    neighbor_mask = (dists < threshold) & ~np.eye(n, dtype=bool)
    diff = y[:, None] != y[None, :]
    neighbor_counts = neighbor_mask.sum(axis=1)
    mismatch_counts = (neighbor_mask & diff).sum(axis=1)
    rho = np.divide(
        mismatch_counts,
        neighbor_counts,
        out=np.zeros(n, dtype=np.float64),
        where=neighbor_counts > 0,
    )

    nu = (1.0 - mu) * rho
    nu = np.clip(nu, 0.0, 1.0)

    s = np.ones(n, dtype=np.float64)
    for i in range(n):
        if nu[i] == 0:
            s[i] = mu[i]
        elif mu[i] <= nu[i]:
            s[i] = 0.0
        else:
            s[i] = (1.0 - nu[i]) / (2.0 - mu[i] - nu[i])
    # Clip to [min_weight, 1.0] so weights stay positive.
    return np.clip(s, min_weight, 1.0)
