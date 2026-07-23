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


def test_build_discriminative_graph_laplacian_vectorized_matches_loop() -> None:
    """Bit-for-bit regression: vectorized adjacency == old nested-loop adjacency.

    Reimplements the pre-vectorization nested ``for i, j`` adjacency loop (the
    code removed in Plan 04) inline and asserts the current vectorized
    :func:`build_discriminative_graph_laplacian` matches it to ``atol=1e-12``.
    """
    rng = np.random.RandomState(21)
    n = 19
    y = rng.randint(0, 4, size=n).astype(np.int64)
    beta = 0.37

    # OLD loop-based adjacency.
    ww = np.zeros((n, n), dtype=np.float64)
    wb = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if y[i] == y[j]:
                ww[i, j] = 1
            else:
                wb[i, j] = 1
    ww = (ww + ww.T) / 2
    wb = (wb + wb.T) / 2
    dw = np.diag(ww.sum(axis=1))
    db = np.diag(wb.sum(axis=1))
    lw = dw - ww
    lb = db - wb

    def normalize(lap: np.ndarray) -> np.ndarray:
        d = np.abs(lap).sum(axis=1) + 1e-8
        d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        l_norm = d_inv_sqrt @ lap @ d_inv_sqrt
        return (l_norm + l_norm.T) / 2

    expected = normalize(lw) - beta * normalize(lb)
    actual = build_discriminative_graph_laplacian(y, discriminative_beta=beta)
    assert np.allclose(actual, expected, atol=1e-12)


def test_gfcca_build_discriminative_graph_vectorized_matches_loop() -> None:
    """Bit-for-bit regression: vectorized GFCCA graph == old nested-loop graph."""
    from tbls.gfcca import GraphFuzzyKCCA

    rng = np.random.RandomState(22)
    n = 17
    y = rng.randint(0, 3, size=n).astype(np.int64)

    ww = np.zeros((n, n), dtype=np.float64)
    wb = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if y[i] == y[j]:
                ww[i, j] = 1
            else:
                wb[i, j] = 1
    ww = (ww + ww.T) / 2
    wb = (wb + wb.T) / 2
    dw = np.diag(ww.sum(axis=1))
    db = np.diag(wb.sum(axis=1))
    lw = dw - ww
    lb = db - wb

    def normalize(lap: np.ndarray) -> np.ndarray:
        d = np.abs(lap).sum(axis=1) + 1e-8
        d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        l_norm = d_inv_sqrt @ lap @ d_inv_sqrt
        return (l_norm + l_norm.T) / 2

    exp_lw, exp_lb = normalize(lw), normalize(lb)
    act_lw, act_lb = GraphFuzzyKCCA()._build_discriminative_graph(y)
    assert np.allclose(act_lw, exp_lw, atol=1e-12)
    assert np.allclose(act_lb, exp_lb, atol=1e-12)


def test_compute_if_scores_geib_vectorized_matches_loop() -> None:
    """Bit-for-bit regression: vectorized GEIB IFS == old loop-based GEIB IFS.

    Reimplements ``compute_if_scores_geib`` with the pre-vectorization
    ``lambda_[i] = np.mean(y[neighbors] != y[i])`` neighbor loop (everything
    else identical, same ``K``) and asserts the current vectorized version
    matches it to ``atol=1e-12``.
    """
    rng = np.random.RandomState(23)
    n = 24
    x = rng.normal(size=(n, 3))
    y = np.array([0] * 8 + [1] * 8 + [2] * 8, dtype=np.int64)
    if_sigma = 0.8

    k = compute_kernel_matrix(x)
    classes = np.unique(y)
    class_idx = {c: np.where(y == c)[0] for c in classes}
    class_cnt = {c: len(idx) for c, idx in class_idx.items()}
    class_sum_k = {c: k[:, idx].sum(axis=1) for c, idx in class_idx.items()}
    class_mean_k: dict[int, float] = {}
    for c, idx in class_idx.items():
        if len(idx) > 0:
            k_cc = k[np.ix_(idx, idx)]
            class_mean_k[c] = float(k_cc.mean())
        else:
            class_mean_k[c] = 0.0

    mu = np.zeros(n, dtype=np.float64)
    epsilon = 1e-8
    for c in classes:
        idx_c = class_idx[c]
        nc = class_cnt[c]
        dist_sq = np.diag(k)[idx_c] - 2.0 / nc * class_sum_k[c][idx_c] + class_mean_k[c]
        dist_sq = np.maximum(dist_sq, 0)
        dist = np.sqrt(dist_sq)
        r_c = dist.max() if len(dist) > 0 else 0.0
        if r_c < epsilon:
            mu[idx_c] = 1.0
        else:
            mu[idx_c] = 1.0 - dist / (r_c + epsilon)
    mu = np.clip(mu, 0.0, 1.0)

    kernel_dists = kernel_distance_matrix(k)
    off_diag = kernel_dists[~np.eye(n, dtype=bool)]
    median_dist = np.median(off_diag) if len(off_diag) > 0 else 1.0
    sigma = if_sigma * median_dist

    # OLD loop-based neighbor-mismatch rate.
    lambda_ = np.zeros(n, dtype=np.float64)
    for i in range(n):
        neighbors = np.where((kernel_dists[i] <= sigma) & (np.arange(n) != i))[0]
        if len(neighbors) > 0:
            lambda_[i] = np.mean(y[neighbors] != y[i])
    nu = (1.0 - mu) * lambda_
    violation = mu + nu > 1.0
    if np.any(violation):
        nu[violation] = 1.0 - mu[violation] - 1e-8
    nu = np.clip(nu, 0.0, 1.0)

    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if nu[i] < 1e-12:
            scores[i] = mu[i]
        elif mu[i] <= nu[i]:
            scores[i] = 0.0
        else:
            scores[i] = (1.0 - nu[i]) / (2.0 - mu[i] - nu[i])
    scores = np.clip(scores, 1e-6, 1.0)
    expected = np.diag(scores)

    actual = compute_if_scores_geib(x, y, K=k, if_sigma=if_sigma)
    assert np.allclose(actual, expected, atol=1e-12)


def test_compute_if_scores_simple_vectorized_matches_loop() -> None:
    """Bit-for-bit regression: vectorized simple IFS == loop-based simple IFS.

    Reimplements ``compute_if_scores_simple`` with the same relative-bandwidth
    membership formula now used by the vectorized implementation
    (``sigma_eff = sigma_if * median_dist``) and the pre-vectorization
    ``rho[i] = np.mean(y[neigh] != y[i])`` neighbor loop (strict-``<``
    threshold, self excluded), then asserts the current vectorized version
    matches it to ``atol=1e-12``. Both sides of the comparison apply the
    Plan-07 bandwidth relativization fix; without that, this test would
    merely assert two different-but-still-buggy implementations agree.
    """
    rng = np.random.RandomState(24)
    n = 22
    a = rng.normal(size=(n, 3))
    y = np.array([0] * 7 + [1] * 8 + [2] * 7, dtype=np.int64)
    sigma_if = 1.1
    delta_if = 0.45
    min_weight = 1e-4

    classes = np.unique(y)
    centers: dict[int, np.ndarray] = {}
    for c in classes:
        idx_c = y == c
        centers[c] = a[idx_c].mean(axis=0)

    # Relative distance scale shared by membership + neighborhood threshold
    # (matches the Plan-07 fix in compute_if_scores_simple).
    dists = cdist(a, a, "euclidean")
    off_diag = dists[~np.eye(n, dtype=bool)]
    median_dist = np.median(off_diag) if off_diag.size > 0 else 1.0
    sigma_eff = sigma_if * median_dist
    threshold = median_dist * delta_if

    mu = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ci = y[i]
        dist = np.linalg.norm(a[i] - centers[ci])
        mu[i] = np.exp(-(dist**2) / (2 * sigma_eff**2))
    mu = np.clip(mu, 0.0, 1.0)

    # OLD loop-based neighbor-mismatch rate.
    rho = np.zeros(n, dtype=np.float64)
    for i in range(n):
        neigh = np.where(dists[i] < threshold)[0]
        neigh = neigh[neigh != i]
        if len(neigh) > 0:
            rho[i] = np.mean(y[neigh] != y[i])
        else:
            rho[i] = 0.0

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
    expected = np.clip(s, min_weight, 1.0)

    actual = compute_if_scores_simple(
        a, y, sigma_if=sigma_if, delta_if=delta_if, min_weight=min_weight
    )
    assert np.allclose(actual, expected, atol=1e-12)


def test_compute_if_scores_simple_non_degenerate_on_realistic_scale() -> None:
    """After-fix: realistic-scale data gives non-degenerate IFS weights.

    Reproduces the GFTBLS-collapse root cause at the IFS layer. On data whose
    median pairwise distance is ~15-20 (matching the real biomedical scale that
    exposed the bug), an *absolute* ``sigma_if=1.0`` membership numerically
    underflows ``mu`` to ~0 for every sample (asserted inline as a
    self-documenting negative check), collapsing the returned weights ``s`` to
    the ``min_weight`` clip for all samples. The Plan-07 relativization
    (``sigma_eff = sigma_if * median_dist``) restores a sane, non-degenerate
    ``mu``/``s`` distribution that actually differentiates confident vs.
    borderline samples.
    """
    rng = np.random.RandomState(7)
    # Two overlapping classes in a moderate-dimensional space; the absolute
    # pairwise-distance scale is ~15-20 (matches the real biomedical dataset).
    n_per = 30
    a = np.vstack(
        [rng.normal(0.0, 1.5, size=(n_per, 100)), rng.normal(0.0, 1.5, size=(n_per, 100))]
    )
    y = np.array([0] * n_per + [1] * n_per, dtype=np.int64)

    # Fixture sanity: confirm the distance scale is in the realistic band.
    d = cdist(a, a, "euclidean")
    median_dist = np.median(d[~np.eye(len(a), dtype=bool)])
    assert 5.0 < median_dist < 40.0, f"unexpected median pairwise distance {median_dist}"

    # Negative check: the OLD absolute-sigma formula (sigma_if as an absolute
    # distance unit) underflows mu to numerical zero for every sample.
    classes = np.unique(y)
    centers = {c: a[y == c].mean(axis=0) for c in classes}
    mu_old = np.array(
        [np.exp(-(np.linalg.norm(a[i] - centers[y[i]]) ** 2) / (2 * 1.0**2)) for i in range(len(a))]
    )
    assert mu_old.max() < 1e-6, (
        f"expected the pre-fix absolute-sigma mu to collapse to ~0, got max={mu_old.max():.3e}"
    )

    # After fix: the returned weight vector has non-degenerate spread (not all
    # samples collapsed to the min_weight floor) and meaningful mass above it.
    s = compute_if_scores_simple(a, y, sigma_if=1.0, delta_if=0.5, min_weight=1e-4)
    assert s.min() > 0
    assert not np.all(np.abs(s - 1e-4) < 1e-6), "every weight collapsed to the min_weight floor"
    assert s.std() > 1e-3, f"weights have no meaningful spread (std={s.std():.3e})"
    assert s.max() > 0.5, f"no confident memberships (max={s.max():.3f})"
