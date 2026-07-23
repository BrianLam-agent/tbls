"""Shared kernel utilities for RBF kernel computation.

Consolidates the three previously duplicated kernel implementations that lived
in ``cca.py``, ``gfcca.py`` and ``tbls.py``.

Cython acceleration candidate: :func:`rbf_kernel` (the
``cdist(X, Y, 'sqeuclidean')`` + ``exp(-gamma * D)`` block is O(n*m*d) and sits
on the hot path of ``TBLS.fit``, ``PairwiseKCCA.fit`` and
``GraphFuzzyKCCA.fit``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]


def rbf_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    gamma: float | None = None,
) -> NDArray[np.float64]:
    """Gaussian RBF kernel matrix with adaptive median-heuristic width.

    The effective kernel width is always adapted by the median pairwise
    distance, so the result is scale-invariant. This merges the identical
    ``rbf_kernel`` bodies previously defined in ``cca.py`` and ``gfcca.py``.

    Args:
        X: First sample matrix of shape ``(n, d)``.
        Y: Second sample matrix of shape ``(m, d)``. If ``None``, ``Y = X``.
        gamma: Base scale. If ``None``, a base scale of ``1.0`` is used. Callers
            that need a specific numeric behavior (e.g. CCA's
            ``DEFAULT_KERNEL_GAMMA = 0.1``) must pass it explicitly; the shared
            default never silently changes a caller's numeric behavior.

    Returns:
        Kernel matrix of shape ``(n, m)``.
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0
    sq_dists: NDArray[np.float64] = cdist(X, Y, "sqeuclidean")
    if gamma == 0.0:
        return X @ Y.T
    all_dists = np.sqrt(sq_dists)
    median_dist = np.median(all_dists) if all_dists.size > 0 else 1.0
    gamma_eff = gamma / (2.0 * median_dist**2) if median_dist > 0 else 1.0
    return np.exp(-gamma_eff * sq_dists)


def compute_kernel_matrix(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """RBF kernel matrix ``K = rbf_kernel(X)`` with adaptive gamma.

    Distinct from :func:`rbf_kernel`: this path computes gamma from the median
    *squared* distance (``1 / (2 * median_sq)``) rather than scaling a base
    gamma by the median distance. It is the variant used by ``TBLS`` for IFS
    score and graph construction.

    Args:
        X: Sample matrix of shape ``(n, d)``.

    Returns:
        Symmetric kernel matrix of shape ``(n, n)``.
    """
    dists_sq: NDArray[np.float64] = cdist(X, X, "sqeuclidean")
    median_sq = np.median(dists_sq[dists_sq > 0]) if np.any(dists_sq > 0) else 1.0
    gamma = 1.0 / (2.0 * median_sq) if median_sq > 0 else 1.0
    return np.exp(-gamma * dists_sq)


def kernel_distance_matrix(K: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pairwise Euclidean distances derived from a kernel matrix.

    ``d(i, j) = sqrt(K(i,i) + K(j,j) - 2*K(i,j))``.

    Args:
        K: Kernel matrix of shape ``(n, n)``.

    Returns:
        Distance matrix of shape ``(n, n)``.
    """
    diag = np.diag(K)[:, None]
    dist_sq = diag + diag.T - 2.0 * K
    dist_sq = np.maximum(dist_sq, 0)
    return np.sqrt(dist_sq)
