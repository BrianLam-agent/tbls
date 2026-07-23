"""Graph Laplacian construction for graph-regularized TBLS.

Extracted from ``tbls.tbls.TBLS._build_graph_laplacian``. Builds the combined
intrinsic/penalty graph Laplacian ``L = alpha_in * L_in - alpha_p * L_p`` used
as a regularizer in :class:`tbls.tbls.TBLS`.

Cython acceleration candidate: the kNN mask + similarity weight construction has
a Python-level loop over edge pairs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]

from . import _kernel


def build_graph_laplacian(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    K: NDArray[np.float64] | None = None,
    graph_alpha_in: float = 1.0,
    graph_alpha_p: float = 1.0,
    graph_knn: int = 10,
    use_kernel: bool = True,
) -> NDArray[np.float64]:
    """Combined intrinsic/penalty graph Laplacian ``L = a_in*L_in - a_p*L_p``.

    Uses normalized Laplacians and (optionally) kernel-space distances.

    Args:
        X: Sample matrix of shape ``(n, d)``.
        y: Integer class labels of shape ``(n,)``.
        K: Precomputed kernel matrix of shape ``(n, n)``. If ``None`` and
            ``use_kernel`` is True, it is computed from ``X``.
        graph_alpha_in: Weight of the intrinsic (within-class) Laplacian.
        graph_alpha_p: Weight of the penalty (between-class) Laplacian.
        graph_knn: Number of nearest neighbors per node. If ``<= 0``, all nodes
            are connected (fully connected graph, no self-loops).
        use_kernel: If True, distances are derived from the kernel matrix;
            otherwise plain Euclidean distances are used.

    Returns:
        Combined Laplacian matrix of shape ``(n, n)``.
    """
    n = X.shape[0]
    if use_kernel:
        if K is None:
            K = _kernel.compute_kernel_matrix(X)
        dists = _kernel.kernel_distance_matrix(K)
    else:
        dists = cdist(X, X, "euclidean")

    # kNN mask
    if graph_knn > 0:
        knn_idx = np.argsort(dists, axis=1)[:, 1 : graph_knn + 1]
        adj = np.zeros((n, n), dtype=bool)
        np.put_along_axis(adj, knn_idx, True, axis=1)
        adj = adj | adj.T
    else:
        adj = np.ones((n, n), dtype=bool)
        np.fill_diagonal(adj, False)

    # Similarity only for the upper triangular part (vectorized)
    i_idx, j_idx = np.where(adj & (np.arange(n)[:, None] < np.arange(n)[None, :]))
    d_vals = dists[i_idx, j_idx]
    # NOTE: the bandwidth is the median over *all* pairwise distances (not just
    # the kNN-selected edges in d_vals) -- this matches the original
    # othercode/tbls.py::_build_graph_laplacian bandwidth exactly. Using only
    # the (already nearest-neighbor-filtered) edge subset would systematically
    # under-estimate the median and change the fitted regularization strength.
    median_dist = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
    eta = np.exp(-(d_vals**2) / (2 * median_dist**2))

    w_in = np.zeros((n, n), dtype=np.float64)
    w_p = np.zeros((n, n), dtype=np.float64)

    class_counts = {c: int(np.sum(y == c)) for c in np.unique(y)}

    for i, j, eta_val in zip(i_idx, j_idx, eta, strict=True):
        if y[i] == y[j]:
            l_c = class_counts[y[i]]
            w_in_val = eta_val / l_c
            w_p_val = eta_val / n * (1.0 - 1.0 / l_c)
            w_in[i, j] = w_in[j, i] = w_in_val
            w_p[i, j] = w_p[j, i] = w_p_val
        else:
            # Different classes: penalty edge only, intrinsic stays 0.
            w_p_val = 1.0 / n
            w_p[i, j] = w_p[j, i] = w_p_val

    # Normalized Laplacians. `np.where` evaluates both branches eagerly, so
    # 1/sqrt(0) triggers a harmless RuntimeWarning for isolated nodes (deg=0)
    # even though that branch's result is discarded; suppress it locally.
    deg_in = w_in.sum(axis=1)
    deg_p = w_p.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        deg_in_inv_sqrt = np.where(deg_in > 0, 1.0 / np.sqrt(deg_in), 0)
        deg_p_inv_sqrt = np.where(deg_p > 0, 1.0 / np.sqrt(deg_p), 0)
    d_inv_sqrt = np.diag(deg_in_inv_sqrt)
    l_in = np.eye(n) - d_inv_sqrt @ w_in @ d_inv_sqrt
    d_p_inv_sqrt = np.diag(deg_p_inv_sqrt)
    l_p = np.eye(n) - d_p_inv_sqrt @ w_p @ d_p_inv_sqrt

    laplacian: NDArray[np.float64] = graph_alpha_in * l_in - graph_alpha_p * l_p
    return laplacian


def build_discriminative_graph_laplacian(
    y: NDArray[np.int64],
    discriminative_beta: float = 0.3,
) -> NDArray[np.float64]:
    """Label-only discriminative graph Laplacian ``L = Lw - beta * Lb``.

    Ported unchanged from
    :meth:`tbls.gfcca.GraphFuzzyKCCA._build_discriminative_graph`: a
    fully-connected (no kNN) same-class/different-class adjacency, each
    Laplacian symmetrically normalized by its own L1-degree
    (``D = diag(abs(L).sum(axis=1) + 1e-8)``) -- a different normalization
    convention than :func:`build_graph_laplacian`'s
    ``I - D^{-1/2} W D^{-1/2}``; kept exactly as GFCCA has it, not reconciled
    with the other convention.

    Args:
        y: Integer class labels of shape ``(n,)``.
        discriminative_beta: Between-class penalty weight.

    Returns:
        Combined Laplacian ``Lw_normalized - beta * Lb_normalized`` of shape
        ``(n, n)``.
    """

    def normalize(lap: NDArray[np.float64]) -> NDArray[np.float64]:
        d = np.abs(lap).sum(axis=1) + 1e-8
        d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        l_norm = d_inv_sqrt @ lap @ d_inv_sqrt
        return (l_norm + l_norm.T) / 2

    # Label-only same/different-class adjacency, vectorized. `same` is
    # symmetric by construction (y[i]==y[j] iff y[j]==y[i]), so the
    # `(x + x.T) / 2` symmetrization below is an exact no-op -- kept for
    # defensive clarity / consistency with the pre-vectorization shape.
    same = (y[:, None] == y[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    diff = 1.0 - same
    np.fill_diagonal(diff, 0.0)
    ww = (same + same.T) / 2
    wb = (diff + diff.T) / 2

    dw = np.diag(ww.sum(axis=1))
    db = np.diag(wb.sum(axis=1))
    lw = dw - ww
    lb = db - wb

    laplacian: NDArray[np.float64] = normalize(lw) - discriminative_beta * normalize(lb)
    return laplacian
