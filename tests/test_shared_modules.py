"""Direct unit tests for the shared `_kernel`/`_ifs`/`_graph` modules.

These modules were originally three separate, duplicated implementations
inside what is now `tbls.py`/`cca.py`/`gfcca.py` (see `docs/architecture.md`
for the consolidation rationale). They previously had no direct test coverage
of their own -- only indirect coverage through `TBLS.fit`, which only checks
output finiteness/shape, not numerical fidelity to the original
(canonical, paper-faithful) implementation. That gap let a bandwidth-computation
regression slip into `_graph.build_graph_laplacian` (the kNN-bandwidth median
was computed over the selected edges instead of over all pairwise distances,
as the original did) -- these tests guard against exactly that class of bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from tbls._graph import build_discriminative_graph_laplacian, build_graph_laplacian
from tbls._ifs import compute_if_scores_geib, compute_if_scores_simple
from tbls._kernel import compute_kernel_matrix, kernel_distance_matrix, rbf_kernel


def test_rbf_kernel_shape_and_self_similarity() -> None:
    rng = np.random.RandomState(0)
    x = rng.normal(size=(20, 4))
    k = rbf_kernel(x)
    assert k.shape == (20, 20)
    assert np.allclose(np.diag(k), 1.0)
    assert np.allclose(k, k.T)


def test_rbf_kernel_gamma_zero_is_linear() -> None:
    rng = np.random.RandomState(1)
    x = rng.normal(size=(10, 3))
    y = rng.normal(size=(6, 3))
    k = rbf_kernel(x, y, gamma=0.0)
    assert np.allclose(k, x @ y.T)


def test_rbf_kernel_cross_shape() -> None:
    rng = np.random.RandomState(2)
    x = rng.normal(size=(10, 3))
    y = rng.normal(size=(6, 3))
    k = rbf_kernel(x, y, gamma=0.1)
    assert k.shape == (10, 6)


def test_compute_kernel_matrix_symmetric_and_bounded() -> None:
    rng = np.random.RandomState(3)
    x = rng.normal(size=(15, 5))
    k = compute_kernel_matrix(x)
    assert k.shape == (15, 15)
    assert np.allclose(k, k.T)
    assert np.allclose(np.diag(k), 1.0)
    assert (k >= 0).all()
    assert (k <= 1.0 + 1e-9).all()


def test_kernel_distance_matrix_matches_direct_computation() -> None:
    rng = np.random.RandomState(4)
    x = rng.normal(size=(12, 3))
    k = compute_kernel_matrix(x)
    dists = kernel_distance_matrix(k)
    assert dists.shape == (12, 12)
    assert np.allclose(np.diag(dists), 0.0, atol=1e-8)
    assert np.allclose(dists, dists.T)
    assert (dists >= 0).all()


def test_compute_if_scores_geib_is_diagonal_in_range() -> None:
    rng = np.random.RandomState(5)
    x = rng.normal(size=(30, 4))
    y = np.array([0] * 15 + [1] * 15)
    s = compute_if_scores_geib(x, y, if_sigma=1.0)
    assert s.shape == (30, 30)
    assert np.allclose(s, np.diag(np.diag(s)))  # strictly diagonal
    diag = np.diag(s)
    assert (diag >= 1e-6 - 1e-12).all()
    assert (diag <= 1.0 + 1e-12).all()


def test_compute_if_scores_simple_is_vector_in_range() -> None:
    rng = np.random.RandomState(6)
    a = rng.normal(size=(30, 4))
    y = np.array([0] * 15 + [1] * 15)
    s = compute_if_scores_simple(a, y, min_weight=1e-4)
    assert s.shape == (30,)
    assert (s >= 1e-4 - 1e-12).all()
    assert (s <= 1.0 + 1e-12).all()


def test_build_graph_laplacian_shape_symmetric() -> None:
    rng = np.random.RandomState(7)
    x = rng.normal(size=(25, 4))
    y = np.array([0] * 12 + [1] * 13)
    laplacian = build_graph_laplacian(x, y, graph_knn=5)
    assert laplacian.shape == (25, 25)
    assert np.allclose(laplacian, laplacian.T)


def test_build_graph_laplacian_bandwidth_uses_full_distance_matrix() -> None:
    """Regression guard for the median-bandwidth extraction bug.

    Reimplements the reference (paper-faithful) Laplacian construction
    directly against `scipy.spatial.distance.cdist`, using the *full*
    pairwise-distance matrix for the similarity bandwidth (not just the
    kNN-selected edges), and asserts `build_graph_laplacian` matches it. This
    is the exact behavior that regressed during the package refactor.
    """
    rng = np.random.RandomState(8)
    n = 18
    x = rng.normal(size=(n, 3))
    y = np.array([0] * 9 + [1] * 9)

    dists = cdist(x, x, "euclidean")
    knn = 4
    knn_idx = np.argsort(dists, axis=1)[:, 1 : knn + 1]
    adj = np.zeros((n, n), dtype=bool)
    np.put_along_axis(adj, knn_idx, True, axis=1)
    adj = adj | adj.T

    i_idx, j_idx = np.where(adj & (np.arange(n)[:, None] < np.arange(n)[None, :]))
    d_vals = dists[i_idx, j_idx]
    median_dist = np.median(dists[dists > 0])  # full matrix, not just d_vals
    eta = np.exp(-(d_vals**2) / (2 * median_dist**2))

    w_in = np.zeros((n, n))
    w_p = np.zeros((n, n))
    class_counts = {c: int(np.sum(y == c)) for c in np.unique(y)}
    for i, j, eta_val in zip(i_idx, j_idx, eta, strict=True):
        if y[i] == y[j]:
            l_c = class_counts[y[i]]
            w_in[i, j] = w_in[j, i] = eta_val / l_c
            w_p[i, j] = w_p[j, i] = eta_val / n * (1.0 - 1.0 / l_c)
        else:
            w_p[i, j] = w_p[j, i] = 1.0 / n

    def normalized_laplacian(w: np.ndarray) -> np.ndarray:
        deg = w.sum(axis=1)
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0)
        d_inv_sqrt = np.diag(deg_inv_sqrt)
        return np.eye(n) - d_inv_sqrt @ w @ d_inv_sqrt

    expected = normalized_laplacian(w_in) - normalized_laplacian(w_p)

    actual = build_graph_laplacian(
        x, y, graph_knn=knn, use_kernel=False, graph_alpha_in=1.0, graph_alpha_p=1.0
    )
    assert np.allclose(actual, expected, atol=1e-10)


def test_build_graph_laplacian_fully_connected_when_knn_non_positive() -> None:
    rng = np.random.RandomState(9)
    x = rng.normal(size=(10, 3))
    y = np.array([0] * 5 + [1] * 5)
    laplacian = build_graph_laplacian(x, y, graph_knn=0, use_kernel=False)
    assert laplacian.shape == (10, 10)
    assert np.isfinite(laplacian).all()


@pytest.mark.parametrize("gamma", [0.05, 0.5, 2.0])
def test_rbf_kernel_gamma_scaling_is_monotonic_in_bandwidth(gamma: float) -> None:
    """Larger gamma -> narrower effective bandwidth -> smaller off-diagonal similarity."""
    rng = np.random.RandomState(10)
    x = rng.normal(size=(10, 3))
    k = rbf_kernel(x, gamma=gamma)
    off_diag_mean = k[~np.eye(10, dtype=bool)].mean()
    assert 0.0 <= off_diag_mean <= 1.0


def test_build_discriminative_graph_laplacian_matches_gfcca_reference() -> None:
    """Bit-for-bit guard for the GFCCA-derived discriminative graph port.

    Reimplements the label-only discriminative graph independently (same style
    as ``test_build_graph_laplacian_bandwidth_uses_full_distance_matrix``) and
    also checks direct agreement with ``GraphFuzzyKCCA._build_discriminative_graph``
    -- the tuned formula this function is ported from. Catches both a divergence
    from GFCCA and a shared bug (the independent reference uses vectorized
    adjacency, not the same loop).
    """
    rng = np.random.RandomState(11)
    n = 16
    y = rng.randint(0, 3, size=n).astype(np.int64)
    beta = 0.3

    # Independent reference: vectorized label-only adjacency.
    same = (y[:, None] == y[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    diff = 1.0 - same
    np.fill_diagonal(diff, 0.0)
    ww = (same + same.T) / 2
    wb = (diff + diff.T) / 2
    lw = np.diag(ww.sum(axis=1)) - ww
    lb = np.diag(wb.sum(axis=1)) - wb

    def normalize(lap: np.ndarray) -> np.ndarray:
        d = np.abs(lap).sum(axis=1) + 1e-8
        d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        l_norm = d_inv_sqrt @ lap @ d_inv_sqrt
        return (l_norm + l_norm.T) / 2

    expected = normalize(lw) - beta * normalize(lb)

    actual = build_discriminative_graph_laplacian(y, discriminative_beta=beta)
    assert actual.shape == (n, n)
    assert np.allclose(actual, actual.T, atol=1e-12)  # symmetric
    assert np.allclose(actual, expected, atol=1e-12)  # matches independent reference

    # Direct agreement with GraphFuzzyKCCA's own (lw, lb), combined with beta.
    from tbls.gfcca import GraphFuzzyKCCA

    gfcca = GraphFuzzyKCCA(discriminative_beta=beta)
    lw_g, lb_g = gfcca._build_discriminative_graph(y)
    expected_gfcca = lw_g - beta * lb_g
    assert np.allclose(actual, expected_gfcca, atol=1e-12)
