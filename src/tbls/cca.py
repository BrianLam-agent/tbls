"""Pairwise regularized kernel CCA for multi-view feature extraction.

Implements paired regularized kernel CCA training and test-time projection, plus
a multi-modal paired-CCA feature-building pipeline. All methods guarantee
zero leakage of test data into training.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

from ._kernel import rbf_kernel

# Global default parameters.
DEFAULT_CCA_K = 7  # canonical variable pairs kept per CCA pair
DEFAULT_CCA_LAMBDA = 0.1  # kernel matrix ridge regularization
DEFAULT_KERNEL_GAMMA = 0.1  # base RBF kernel gamma


class PairwiseKCCA(BaseEstimator):  # type: ignore[misc]
    """Paired regularized kernel CCA between two views.

    Finds projection directions maximizing correlation in the kernel space of
    two views and can project new samples onto them.

    This is a two-view estimator: :meth:`fit` takes ``X1`` and ``X2``, and
    projection is done via :meth:`transform_view1` / :meth:`transform_view2`.
    The no-argument :meth:`transform` returns the training projections of both
    views. It deliberately does **not** inherit ``TransformerMixin`` because the
    two-view API does not match sklearn's single-argument ``transform(X)``
    contract (see ``docs/architecture.md`` and ``docs/usage-cca-gfcca.md``).

    Args:
        k: Number of canonical variable pairs to keep.
        reg_lambda: Kernel matrix regularization coefficient (ridge penalty).
        kernel_gamma: ``gamma`` passed to :func:`tbls._kernel.rbf_kernel`.
    """

    def __init__(
        self,
        k: int = DEFAULT_CCA_K,
        reg_lambda: float = DEFAULT_CCA_LAMBDA,
        kernel_gamma: float = DEFAULT_KERNEL_GAMMA,
    ) -> None:
        self.k = k
        self.reg_lambda = reg_lambda
        self.kernel_gamma = kernel_gamma

        # Fitted attributes (assigned in fit).
        self.X_train1_: NDArray[np.float64] | None = None
        self.X_train2_: NDArray[np.float64] | None = None
        self.alpha1_: NDArray[np.float64] | None = None  # view-1 projections (n_train, k)
        self.alpha2_: NDArray[np.float64] | None = None  # view-2 projections (n_train, k)

    def fit(self, X1: NDArray[np.float64], X2: NDArray[np.float64]) -> PairwiseKCCA:
        """Fit the kernel CCA model on training data from two views.

        Args:
            X1: View-1 training data of shape ``(n_samples, n_features_1)``.
            X2: View-2 training data of shape ``(n_samples, n_features_2)``.

        Returns:
            The fitted instance.
        """
        k1 = rbf_kernel(X1, gamma=self.kernel_gamma)
        k2 = rbf_kernel(X2, gamma=self.kernel_gamma)
        n = k1.shape[0]
        lam = self.reg_lambda
        identity = np.eye(n)

        # Generalized eigenvalue problem matrices.
        a_mat = np.block(
            [
                [np.zeros((n, n)), k1 @ k2.T],
                [k2 @ k1.T, np.zeros((n, n))],
            ]
        )
        b_mat = np.block(
            [
                [k1 @ k1.T + lam * identity, np.zeros((n, n))],
                [np.zeros((n, n)), k2 @ k2.T + lam * identity],
            ]
        )

        eigvals, eigvecs = eigh(a_mat, b_mat)
        idx = np.argsort(eigvals)[-self.k :]
        alphas = eigvecs[:, idx]  # shape (2N, k)
        self.alpha1_ = alphas[:n, :]  # (N, k)
        self.alpha2_ = alphas[n:, :]  # (N, k)

        # Keep training data for test-time projection.
        self.X_train1_ = X1
        self.X_train2_ = X2
        return self

    def transform_view1(self, X_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project new samples onto view-1's CCA space.

        Args:
            X_new: View-1 data of shape ``(m_samples, n_features_1)``.

        Returns:
            Projected features of shape ``(m_samples, k)``.
        """
        if self.alpha1_ is None or self.X_train1_ is None:
            raise RuntimeError("Model not fitted; call fit() first.")
        x_train1 = self.X_train1_
        alpha1 = self.alpha1_
        k_mat = rbf_kernel(X_new, x_train1, gamma=self.kernel_gamma)
        return k_mat @ alpha1

    def transform_view2(self, X_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project new samples onto view-2's CCA space.

        Args:
            X_new: View-2 data of shape ``(m_samples, n_features_2)``.

        Returns:
            Projected features of shape ``(m_samples, k)``.
        """
        if self.alpha2_ is None or self.X_train2_ is None:
            raise RuntimeError("Model not fitted; call fit() first.")
        x_train2 = self.X_train2_
        alpha2 = self.alpha2_
        k_mat = rbf_kernel(X_new, x_train2, gamma=self.kernel_gamma)
        return k_mat @ alpha2

    def transform(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the training projections of both views.

        Mainly used by :func:`build_cca_features` to obtain training features.

        Returns:
            A pair ``(Z1, Z2)`` of shape ``(n_train, k)`` each.
        """
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
        z1 = k1 @ alpha1
        z2 = k2 @ alpha2
        return z1, z2


def build_cca_features(
    X_views: list[NDArray[np.float64]],
    cca_k: int = DEFAULT_CCA_K,
    cca_lambda: float = DEFAULT_CCA_LAMBDA,
    kernel_gamma: float = DEFAULT_KERNEL_GAMMA,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], PairwiseKCCA]]:
    """Run pairwise CCA over views and concatenate canonical variables.

    For each view pair ``(i, j)`` a :class:`PairwiseKCCA` is trained and both
    views' projection vectors are extracted. All projections are concatenated
    along the feature axis into a single feature matrix.

    Args:
        X_views: List of view matrices, each of shape ``(n_samples, n_features_i)``.
        cca_k: Canonical variable pairs kept per CCA pair.
        cca_lambda: Kernel matrix regularization coefficient.
        kernel_gamma: Kernel ``gamma`` parameter.

    Returns:
        A pair ``(F_train, cca_models)`` where ``F_train`` is the concatenated
        training feature matrix and ``cca_models`` maps ``(i, j)`` view pairs to
        the fitted :class:`PairwiseKCCA` for later projection of new data.
    """
    n_views = len(X_views)
    feature_blocks: list[NDArray[np.float64]] = []
    cca_models: dict[tuple[int, int], PairwiseKCCA] = {}

    for i in range(n_views):
        for j in range(i + 1, n_views):
            cca = PairwiseKCCA(k=cca_k, reg_lambda=cca_lambda, kernel_gamma=kernel_gamma)
            cca.fit(X_views[i], X_views[j])
            z_i, z_j = cca.transform()  # training projections of both views
            feature_blocks.append(z_i)
            feature_blocks.append(z_j)
            cca_models[(i, j)] = cca

    f_train = np.hstack(feature_blocks)
    return f_train, cca_models


def project_cca_features(
    X_views_new: list[NDArray[np.float64]],
    cca_models: dict[tuple[int, int], PairwiseKCCA],
) -> NDArray[np.float64]:
    """Project new data into the fused CCA feature matrix.

    Args:
        X_views_new: Test-time views, in the same order as at training time.
        cca_models: Model dict returned by :func:`build_cca_features`.

    Returns:
        Projected fused feature matrix of shape ``(n_samples, total_features)``.
    """
    blocks: list[NDArray[np.float64]] = []
    for (i, j), cca in cca_models.items():
        z_i = cca.transform_view1(X_views_new[i])
        z_j = cca.transform_view2(X_views_new[j])
        blocks.append(z_i)
        blocks.append(z_j)
    return np.hstack(blocks)
