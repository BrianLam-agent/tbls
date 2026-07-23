"""Broad Learning System (BLS) for multi-class classification.

A sklearn-compatible BLS estimator with random weight mapping, enhancement
nodes, ridge pseudoinverse learning, optional class weighting and Woodbury-form
incremental enhancement.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, ClassifierMixin  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from sklearn.utils.class_weight import compute_class_weight  # type: ignore[import-untyped]

Activation = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class BroadLearningSystem(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Broad Learning System with ridge pseudoinverse learning.

    Args:
        n_feature_groups: Number of feature mapping groups.
        n_feature_nodes_per_group: Number of nodes in each feature mapping
            group.
        n_enhancement_groups: Number of enhancement groups.
        n_enhancement_nodes_per_group: Number of nodes in each enhancement
            group.
        map_func: Activation for feature mapping. One of ``'relu'``,
            ``'sigmoid'``, ``'tanh'``, ``'linear'``, ``'leaky_relu'``.
        enhance_func: Activation for enhancement nodes.
        reg_param: Ridge regularization parameter.
        class_weights: If ``'auto'``, class weights are computed as inversely
            proportional to class frequencies. If a dict, must be
            ``{class_label: weight}``. If ``None``, no weighting.
        random_state: Seed for random number generation.
    """

    def __init__(
        self,
        n_feature_groups: int = 10,
        n_feature_nodes_per_group: int = 100,
        n_enhancement_groups: int = 10,
        n_enhancement_nodes_per_group: int = 100,
        map_func: str = "relu",
        enhance_func: str = "relu",
        reg_param: float = 1e-8,
        class_weights: str | dict[Any, Any] | None = None,
        random_state: int | None = None,
    ) -> None:
        self.n_feature_groups = n_feature_groups
        self.n_feature_nodes_per_group = n_feature_nodes_per_group
        self.n_enhancement_groups = n_enhancement_groups
        self.n_enhancement_nodes_per_group = n_enhancement_nodes_per_group
        self.map_func = map_func
        self.enhance_func = enhance_func
        self.reg_param = reg_param
        self.class_weights = class_weights
        self.random_state = random_state

    def _get_activation(self, name: str) -> Activation:
        activations: dict[str, Activation] = {
            "relu": lambda x: np.maximum(0, x),
            "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
            "tanh": np.tanh,
            "linear": lambda x: x,
            "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation: {name}")
        return activations[name]

    def _generate_weights(
        self, in_dim: int, out_dim: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Generate random weights and biases uniformly from ``[-1, 1]``."""
        rng = np.random.RandomState(self.random_state)
        weights = rng.uniform(-1, 1, size=(in_dim, out_dim))
        biases = rng.uniform(-1, 1, size=(1, out_dim))
        return weights, biases

    def _one_hot_encode(self, y: NDArray[np.int64]) -> NDArray[np.float64]:
        """Convert integer labels to a one-hot encoded matrix."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return np.eye(self.n_classes_, dtype=np.float64)[y]

    def _compute_sample_weights(
        self, y: NDArray[np.int64]
    ) -> NDArray[np.float64] | None:
        """Compute sample weights based on the ``class_weights`` parameter."""
        if self.class_weights is None:
            return None
        if isinstance(self.class_weights, dict):
            weights = np.ones_like(y, dtype=np.float64)
            for cls, weight in self.class_weights.items():
                weights[y == cls] = weight
            return weights
        if self.class_weights == "auto":
            # ``compute_class_weight`` returns one weight per class.
            class_weights_array: NDArray[np.float64] = compute_class_weight(
                class_weight="balanced", classes=self.classes_, y=y
            )
            return class_weights_array[y]
        raise ValueError(
            f"Unsupported class_weights value: {self.class_weights}. "
            "Expected None, 'auto', or dict."
        )

    def _build_feature_nodes(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Construct feature mapping nodes using random weights and activation."""
        self._map_weights: list[NDArray[np.float64]] = []
        self._map_biases: list[NDArray[np.float64]] = []
        z_parts = []
        for _ in range(self.n_feature_groups):
            weights, biases = self._generate_weights(
                X.shape[1], self.n_feature_nodes_per_group
            )
            self._map_weights.append(weights)
            self._map_biases.append(biases)
            z_parts.append(self._act_map(X @ weights + biases))
        return np.hstack(z_parts)

    def _build_enhancement_nodes(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Construct enhancement nodes based on feature nodes."""
        self._enh_weights: list[NDArray[np.float64]] = []
        self._enh_biases: list[NDArray[np.float64]] = []
        h_parts = []
        for _ in range(self.n_enhancement_groups):
            weights, biases = self._generate_weights(
                Z.shape[1], self.n_enhancement_nodes_per_group
            )
            self._enh_weights.append(weights)
            self._enh_biases.append(biases)
            h_parts.append(self._act_enh(Z @ weights + biases))
        return np.hstack(h_parts)

    def fit(
        self, X: NDArray[np.float64], y: NDArray[np.int64]
    ) -> "BroadLearningSystem":
        """Train the BLS model.

        Args:
            X: Training samples of shape ``(n_samples, n_features)``.
            y: Target labels of shape ``(n_samples,)``.

        Returns:
            The fitted estimator (self).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        x_clean = np.nan_to_num(X, nan=0.0)
        y_clean = np.nan_to_num(y.ravel(), nan=0.0).astype(np.int64)

        # Standardization
        self.scaler_ = StandardScaler()
        x_scaled = self.scaler_.fit_transform(x_clean)

        # One-hot encoding for multi-class
        y_onehot = self._one_hot_encode(y_clean)

        # Activation functions
        self._act_map = self._get_activation(self.map_func)
        self._act_enh = self._get_activation(self.enhance_func)

        # Build feature and enhancement nodes
        z = self._build_feature_nodes(x_scaled)
        h = self._build_enhancement_nodes(z)
        a = np.hstack([z, h])  # combined input matrix

        # Pseudoinverse with optional sample weighting
        sample_weights = self._compute_sample_weights(y_clean)
        if sample_weights is not None:
            # Weighted ridge regression
            diag = np.diag(sample_weights)
            a_pinv = np.linalg.pinv(
                a.T @ diag @ a + self.reg_param * np.eye(a.shape[1])
            ) @ a.T @ diag
        else:
            a_pinv = np.linalg.pinv(
                a.T @ a + self.reg_param * np.eye(a.shape[1])
            ) @ a.T

        # Output weights
        self.W_ = a_pinv @ y_onehot

        # Keep components needed for prediction / incremental learning
        self.A_train_ = a
        self.Y_train_onehot_ = y_onehot
        self.sample_weights_ = sample_weights  # may be None

        self.fitted_ = True
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict class labels."""
        if not self.fitted_:
            raise RuntimeError("Model not fitted yet.")
        x_clean = np.nan_to_num(X, nan=0.0)
        x_scaled = self.scaler_.transform(x_clean)
        raw = self._compute_raw_output(x_scaled)
        return np.argmax(raw, axis=1).astype(np.int64)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float32]:
        """Predict class probabilities."""
        if not self.fitted_:
            raise RuntimeError("Model not fitted yet.")
        x_clean = np.nan_to_num(X, nan=0.0)
        x_scaled = self.scaler_.transform(x_clean)
        raw = self._compute_raw_output(x_scaled)
        return np.asarray(softmax(raw, axis=1), dtype=np.float32)

    def _compute_raw_output(self, X_scaled: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the linear output before softmax."""
        z = self._recompute_feature_nodes(X_scaled)
        h = self._recompute_enhancement_nodes(z)
        a = np.hstack([z, h])
        return a @ self.W_

    def _recompute_feature_nodes(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct feature nodes using stored weights."""
        z_parts = [
            self._act_map(X @ weights + biases)
            for weights, biases in zip(self._map_weights, self._map_biases, strict=True)
        ]
        return np.hstack(z_parts)

    def _recompute_enhancement_nodes(
        self, Z: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Reconstruct enhancement nodes using stored weights."""
        h_parts = [
            self._act_enh(Z @ weights + biases)
            for weights, biases in zip(self._enh_weights, self._enh_biases, strict=True)
        ]
        return np.hstack(h_parts)

    def incremental_enhance(
        self, X: NDArray[np.float64], num_new_nodes: int = 100
    ) -> "BroadLearningSystem":
        """Incrementally add one enhancement group with ``num_new_nodes`` nodes.

        Uses the Woodbury formula to update the pseudo-inverse without retraining
        the whole network. ``X`` should be the same training data used in
        :meth:`fit` (or a new batch for which enhancement nodes are to be added).

        Args:
            X: Samples of shape ``(n_samples, n_features)``.
            num_new_nodes: Number of new enhancement nodes to add.

        Returns:
            The updated estimator (self).
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first.")

        x_clean = np.nan_to_num(X, nan=0.0)
        x_scaled = self.scaler_.transform(x_clean)

        # Original feature and enhancement nodes for the current training data
        z = self._recompute_feature_nodes(x_scaled)
        h_old = self._recompute_enhancement_nodes(z)
        a_old = np.hstack([z, h_old])

        # Generate new enhancement nodes
        weights_new, biases_new = self._generate_weights(z.shape[1], num_new_nodes)
        self._enh_weights.append(weights_new)
        self._enh_biases.append(biases_new)
        h_new = self._act_enh(z @ weights_new + biases_new)

        # Compute the initial pseudo-inverse if not already stored.
        if not hasattr(self, "A_pinv_"):
            if self.sample_weights_ is not None:
                diag = np.diag(self.sample_weights_)
                self.A_pinv_ = np.linalg.pinv(
                    a_old.T @ diag @ a_old + self.reg_param * np.eye(a_old.shape[1])
                ) @ a_old.T @ diag
            else:
                self.A_pinv_ = np.linalg.pinv(
                    a_old.T @ a_old + self.reg_param * np.eye(a_old.shape[1])
                ) @ a_old.T

        d_mat = self.A_pinv_ @ h_new
        c_mat = h_new - a_old @ d_mat
        b_mat = np.linalg.pinv(c_mat.T @ c_mat + self.reg_param * np.eye(c_mat.shape[1])) @ c_mat.T
        self.A_pinv_ = np.vstack([self.A_pinv_ - d_mat @ b_mat, b_mat])

        # Update output weights
        self.W_ = self.A_pinv_ @ self.Y_train_onehot_
        return self
