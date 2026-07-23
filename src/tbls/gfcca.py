"""Graph-embedded intuitionistic-fuzzy kernel CCA (Graph-Fuzzy KCCA).

Extends regularized kernel CCA with:

- Intuitionistic fuzzy scores: per-sample credibility clipped to a positive
  range to avoid zero weights.
- Graph-embedding regularization: a discriminative graph (``Lw - beta * Lb``)
  with a stability term to keep the generalized matrix numerically positive
  definite.

The interface is fully compatible with :mod:`tbls.cca`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError as ScipyLinAlgError, eigh  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

from ._ifs import compute_if_scores_simple
from ._kernel import rbf_kernel


class GraphFuzzyKCCA(BaseEstimator):  # type: ignore[misc]
    """Graph-embedded intuitionistic-fuzzy kernel CCA model.

    Two-view estimator (like :class:`tbls.cca.PairwiseKCCA`); does not inherit
    ``TransformerMixin`` for the same reason (see ``docs/architecture.md`` and
    ``docs/usage-cca-gfcca.md``).

    Args:
        k: Number of canonical variable pairs to keep.
        reg_lambda: Kernel matrix regularization coefficient.
        kernel_gamma: ``gamma`` passed to :func:`tbls._kernel.rbf_kernel`.
        graph_gamma: Weight of the graph regularization term.
        sigma_if: Gaussian width for IFS membership computation.
        delta_if: Relative distance threshold for the IFS neighborhood
            (multiplied by the median pairwise Euclidean distance).
        min_weight: Minimum IFS weight clip; prevents zero weights that would
            make the regularized matrix singular.
        discriminative_beta: Between-class penalty weight in the discriminative
            graph.
        graph_tau: Reserved stability term (currently unused at runtime).
        epsilon_B: Small global identity added to guarantee ``B`` is positive
            definite.
    """

    def __init__(
        self,
        k: int = 5,
        reg_lambda: float = 0.1,
        kernel_gamma: float = 1.0,
        graph_gamma: float = 0.1,
        sigma_if: float = 1.0,
        delta_if: float = 0.5,
        min_weight: float = 1e-4,
        discriminative_beta: float = 0.3,
        graph_tau: float = 1e-3,
        epsilon_B: float = 1e-6,
    ) -> None:
        self.k = k
        self.reg_lambda = reg_lambda
        self.kernel_gamma = kernel_gamma
        self.graph_gamma = graph_gamma
        self.sigma_if = sigma_if
        self.delta_if = delta_if
        self.min_weight = min_weight
        self.discriminative_beta = discriminative_beta
        self.graph_tau = graph_tau
        self.epsilon_B = epsilon_B

        self.X_train1_: NDArray[np.float64] | None = None
        self.X_train2_: NDArray[np.float64] | None = None
        self.alpha1_: NDArray[np.float64] | None = None
        self.alpha2_: NDArray[np.float64] | None = None

    def _center_kernel(self, K: NDArray[np.float64]) -> NDArray[np.float64]:
        """Center a kernel matrix: ``H K H`` where ``H = I - 1/n * 1 1^T``."""
        n = K.shape[0]
        h_mat = np.eye(n) - np.ones((n, n)) / n
        centered: NDArray[np.float64] = h_mat @ K @ h_mat
        return centered

    def _compute_if_scores(
        self, A: NDArray[np.float64], y: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Intuitionistic fuzzy score vector, clipped to ``[min_weight, 1]``."""
        return compute_if_scores_simple(
            A,
            y,
            sigma_if=self.sigma_if,
            delta_if=self.delta_if,
            min_weight=self.min_weight,
        )

    def _build_discriminative_graph(
        self, y: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build the discriminative graph, returning symmetric-normalized ``(Lw, Lb)``."""

        def normalize(lap: NDArray[np.float64]) -> NDArray[np.float64]:
            d = np.abs(lap).sum(axis=1) + 1e-8
            d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
            l_norm = d_inv_sqrt @ lap @ d_inv_sqrt
            return (l_norm + l_norm.T) / 2

        n = len(y)
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
        # Symmetrize.
        ww = (ww + ww.T) / 2
        wb = (wb + wb.T) / 2

        dw = np.diag(ww.sum(axis=1))
        db = np.diag(wb.sum(axis=1))
        lw = dw - ww
        lb = db - wb

        return normalize(lw), normalize(lb)

    def fit(
        self,
        X1: NDArray[np.float64],
        X2: NDArray[np.float64],
        y: NDArray[np.int64],
        max_attempts: int = 5,
        tau_factor: float = 10.0,
    ) -> GraphFuzzyKCCA:
        """Fit the model, increasing regularization on non-PD errors.

        Args:
            X1: View-1 training data of shape ``(n_samples, n_features_1)``.
            X2: View-2 training data of shape ``(n_samples, n_features_2)``.
            y: Integer class labels of shape ``(n_samples,)``.
            max_attempts: Maximum number of retries.
            tau_factor: Amplification factor for ``epsilon_B`` on each retry
                (``reg_lambda`` is not amplified).

        Returns:
            The fitted instance.
        """
        n = X1.shape[0]
        k1 = rbf_kernel(X1, gamma=self.kernel_gamma)
        k2 = rbf_kernel(X2, gamma=self.kernel_gamma)

        # Center kernels.
        k1 = self._center_kernel(k1)
        k2 = self._center_kernel(k2)

        s1 = self._compute_if_scores(X1, y)
        s2 = self._compute_if_scores(X2, y)
        s = (s1 + s2) / 2.0
        w_mat = np.diag(s)

        wk1 = w_mat @ k1
        wk2 = w_mat @ k2

        lw, lb = self._build_discriminative_graph(y)
        beta = self.discriminative_beta
        graph_term1 = (k1 @ lw @ k1.T) - beta * (k1 @ lb @ k1.T)
        graph_term2 = (k2 @ lw @ k2.T) - beta * (k2 @ lb @ k2.T)

        lam = self.reg_lambda
        g_g = self.graph_gamma
        eps = self.epsilon_B

        eigvals: NDArray[np.float64] = np.array([])
        eigvecs: NDArray[np.float64] = np.array([])
        for attempt in range(max_attempts):
            a_mat = np.block(
                [
                    [np.zeros((n, n)), wk1 @ wk2.T],
                    [wk2 @ wk1.T, np.zeros((n, n))],
                ]
            )
            b11 = wk1 @ wk1.T + lam * np.eye(n) + g_g * graph_term1
            b22 = wk2 @ wk2.T + lam * np.eye(n) + g_g * graph_term2
            b_mat = np.block(
                [
                    [b11, np.zeros((n, n))],
                    [np.zeros((n, n)), b22],
                ]
            )

            # Symmetrize.
            a_mat = (a_mat + a_mat.T) / 2
            b_mat = (b_mat + b_mat.T) / 2

            try:
                eigvals_b = np.linalg.eigvalsh(b_mat)
                min_eig_b = float(np.min(eigvals_b))
            except np.linalg.LinAlgError:
                min_eig_b = -1.0
            if min_eig_b <= 0:
                b_mat += (abs(min_eig_b) + eps) * np.eye(2 * n)
            else:
                b_mat += eps * np.eye(2 * n)

            try:
                eigvals, eigvecs = eigh(a_mat, b_mat)
            except (np.linalg.LinAlgError, ScipyLinAlgError):
                if attempt == max_attempts - 1:
                    raise
                eps *= tau_factor
                continue
            break

        idx = np.argsort(eigvals)[-self.k :]
        alphas = eigvecs[:, idx]
        self.alpha1_ = alphas[:n, :]
        self.alpha2_ = alphas[n:, :]

        self.X_train1_ = X1
        self.X_train2_ = X2
        return self

    def transform_view1(self, X_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project new samples onto view-1."""
        if self.alpha1_ is None or self.X_train1_ is None:
            raise RuntimeError("Model not fitted; call fit() first.")
        x_train1 = self.X_train1_
        alpha1 = self.alpha1_
        k_mat = rbf_kernel(X_new, x_train1, gamma=self.kernel_gamma)
        # Test-time centering is intentionally omitted for simplicity.
        return k_mat @ alpha1

    def transform_view2(self, X_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project new samples onto view-2."""
        if self.alpha2_ is None or self.X_train2_ is None:
            raise RuntimeError("Model not fitted; call fit() first.")
        x_train2 = self.X_train2_
        alpha2 = self.alpha2_
        k_mat = rbf_kernel(X_new, x_train2, gamma=self.kernel_gamma)
        return k_mat @ alpha2

    def transform(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the training projections of both views."""
        if (
            self.alpha1_ is None
            or self.alpha2_ is None
            or self.X_train1_ is None
            or self.X_train2_ is None
        ):
            raise RuntimeError("Model not fitted; call fit() first.")
        x_train1 = self.X_train1_
        x_train2 = self.X_train2_
        alpha1 = self.alpha1_
        alpha2 = self.alpha2_
        k1 = rbf_kernel(x_train1, gamma=self.kernel_gamma)
        k2 = rbf_kernel(x_train2, gamma=self.kernel_gamma)
        # Training kernels were centered in fit; no re-centering needed here.
        z1 = k1 @ alpha1
        z2 = k2 @ alpha2
        return z1, z2


def build_gfcca_features(
    X_views: list[NDArray[np.float64]],
    y: NDArray[np.int64],
    cca_k: int = 7,
    cca_lambda: float = 0.1,
    kernel_gamma: float = 0.1,
    sigma_graph: float | None = None,
    graph_gamma: float = 0.5,
    sigma_if: float = 1.0,
    delta_if: float = 0.5,
    min_weight: float = 1e-4,
    discriminative_beta: float = 0.3,
    graph_tau: float = 1e-3,
    epsilon_B: float = 1e-6,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], GraphFuzzyKCCA]]:
    """Multi-view paired graph-fuzzy CCA returning a concatenated feature matrix.

    Args:
        X_views: List of view matrices, each ``(n_samples, n_features_i)``.
        y: Integer class labels of shape ``(n_samples,)``.
        cca_k: Canonical variable pairs kept per CCA pair.
        cca_lambda: Kernel matrix regularization coefficient.
        kernel_gamma: Kernel ``gamma`` parameter.
        sigma_graph: Reserved (unused).
        graph_gamma: Weight of the graph regularization term.
        sigma_if: Gaussian width for IFS membership.
        delta_if: Relative IFS neighborhood threshold.
        min_weight: Minimum IFS weight clip.
        discriminative_beta: Between-class penalty weight.
        graph_tau: Reserved stability term.
        epsilon_B: Small global identity for PD guarantee.

    Returns:
        A pair ``(F_train, gfcca_models)`` of the concatenated training feature
        matrix and a ``(i, j)`` -> :class:`GraphFuzzyKCCA` model dict.
    """
    _ = sigma_graph  # reserved for API compatibility
    n_views = len(X_views)
    feature_blocks: list[NDArray[np.float64]] = []
    gfcca_models: dict[tuple[int, int], GraphFuzzyKCCA] = {}

    for i in range(n_views):
        for j in range(i + 1, n_views):
            model = GraphFuzzyKCCA(
                k=cca_k,
                reg_lambda=cca_lambda,
                kernel_gamma=kernel_gamma,
                graph_gamma=graph_gamma,
                sigma_if=sigma_if,
                delta_if=delta_if,
                min_weight=min_weight,
                discriminative_beta=discriminative_beta,
                graph_tau=graph_tau,
                epsilon_B=epsilon_B,
            )
            model.fit(X_views[i], X_views[j], y)
            z_i, z_j = model.transform()
            feature_blocks.append(z_i)
            feature_blocks.append(z_j)
            gfcca_models[(i, j)] = model

    f_train = np.hstack(feature_blocks)
    return f_train, gfcca_models


def project_cca_features(
    X_views_new: list[NDArray[np.float64]],
    cca_models: dict[tuple[int, int], GraphFuzzyKCCA],
) -> NDArray[np.float64]:
    """Project new data; same interface as :func:`tbls.cca.project_cca_features`."""
    blocks: list[NDArray[np.float64]] = []
    for (i, j), model in cca_models.items():
        z_i = model.transform_view1(X_views_new[i])
        z_j = model.transform_view2(X_views_new[j])
        blocks.append(z_i)
        blocks.append(z_j)
    return np.hstack(blocks)
