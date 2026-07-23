"""Tree-based Broad Learning System (TBLS) for multi-class classification.

Features:
    - Regression trees (each tree outputs a scalar leaf mean).
    - Bootstrap sampling with Poisson(1) (paper's original scheme).
    - Random Subspace Method (RSM).
    - Intuitionistic Fuzzy Set (IFS) scores computed in kernel space.
    - Graph embedding (intrinsic & penalty graphs) with normalized Laplacian.
    - Kernel matrix reused across IFS and graph.
    - Fully vectorized graph construction.
    - sklearn-compatible ``random_state`` (int or None).
    - Poisson bootstrapping with sample-size guarantee.
    - Incremental layers (recompute weights for stability).

Kernel, IFS and graph-Laplacian helpers live in the sibling private modules
:mod:`tbls._kernel`, :mod:`tbls._ifs` and :mod:`tbls._graph`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import pinv  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, ClassifierMixin  # type: ignore[import-untyped]
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeRegressor  # type: ignore[import-untyped]

from . import _graph, _ifs, _kernel


class RegressionTreeModule:
    """Single regression tree with Poisson bootstrap and RSM.

    Args:
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples to split a node.
        max_features_ratio: Fraction of features sampled per tree (RSM).
        random_state: Seed. If not None, a per-tree seed is derived from the
            shared RNG; if None the tree uses its own randomness.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 3,
        max_features_ratio: float = 0.7,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_ratio = max_features_ratio
        self.random_state = random_state
        self.tree: DecisionTreeRegressor | None = None
        self.feature_indices_: NDArray[np.intp] | None = None

    def fit(
        self, X: NDArray[np.float64], y: NDArray[np.int64], rng: np.random.RandomState
    ) -> RegressionTreeModule:
        """Fit the tree on a Poisson-bootstrap, random-subspace view of ``X``."""
        n_samples, n_features = X.shape
        # Poisson(1) bootstrap
        counts = rng.poisson(1.0, size=n_samples)
        indices = np.repeat(np.arange(n_samples), counts)
        # Guarantee enough samples (at least min_samples_split)
        if len(indices) < self.min_samples_split:
            extra = rng.choice(n_samples, size=self.min_samples_split - len(indices), replace=True)
            indices = np.concatenate([indices, extra])
        x_boot = X[indices]
        y_boot = y[indices]

        # Random Subspace Method
        n_selected = max(1, int(n_features * self.max_features_ratio))
        self.feature_indices_ = rng.choice(n_features, size=n_selected, replace=False)
        x_boot_rsm = x_boot[:, self.feature_indices_]

        # Decision tree regressor
        if self.random_state is not None:
            tree_seed: int | None = int(rng.randint(0, 2**31))
        else:
            tree_seed = None
        self.tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=tree_seed,
        )
        self.tree.fit(x_boot_rsm, y_boot)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict scalar outputs, reshaped to ``(n, 1)``."""
        if self.feature_indices_ is None or self.tree is None:
            raise RuntimeError("Tree module not fitted.")
        x_rsm = X[:, self.feature_indices_]
        return np.asarray(self.tree.predict(x_rsm), dtype=np.float64).reshape(-1, 1)


class TBLS(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Tree Broad Learning System with IFS, graph regularization and incremental layers.

    Args:
        n_map_trees: Number of mapping (feature) trees.
        n_enhance_trees: Number of enhancement trees.
        n_increment_layers: Number of incremental enhancement layers added
            after the initial fit.
        tree_max_depth: Maximum depth of each regression tree.
        tree_min_samples_split: Minimum samples to split a tree node.
        tree_max_features_ratio: RSM feature fraction per tree.
        reg_param: Ridge regularization parameter.
        use_if_weights: If True, weight samples by IFS credibility scores
            (formula selected by ``if_strategy``; default ``"simple"``).
        if_sigma: Scaling factor for the IFS neighborhood radius / membership width.
        graph_gamma: Weight of graph regularization (0 disables it).
        graph_alpha_in: Weight of the intrinsic (within-class) Laplacian (kNN graph only).
        graph_alpha_p: Weight of the penalty (between-class) Laplacian (kNN graph only).
        graph_knn: Number of nearest neighbors in the kNN graph.
        graph_threshold: Reserved threshold hyperparameter (unused currently).
        class_sensitive: Reserved class-sensitive flag (unused currently).
        random_state: Seed for reproducibility.
        use_kernel_for_graph: If True, kNN-graph distances are kernel-space.
        graph_strategy: Graph-Laplacian formula. ``"discriminative"`` (default)
            uses GraphFuzzyKCCA's tuned label-only discriminative graph
            (``Lw - beta * Lb``); ``"knn"`` reproduces the estimator's original
            kNN-graph behavior unchanged.
        if_strategy: IFS scoring formula. ``"simple"`` (default) uses
            GraphFuzzyKCCA's tuned per-class-center + relative-neighborhood
            formula; ``"geib"`` reproduces the original GEIB formulation.
        discriminative_beta: Between-class penalty weight for the discriminative
            graph (``graph_strategy="discriminative"``).
        if_delta: Relative neighborhood threshold for the simple IFS formula
            (``if_strategy="simple"``).
        if_min_weight: Minimum IFS weight clip for the simple IFS formula.
    """

    def __init__(
        self,
        n_map_trees: int = 20,
        n_enhance_trees: int = 20,
        n_increment_layers: int = 0,
        tree_max_depth: int = 5,
        tree_min_samples_split: int = 3,
        tree_max_features_ratio: float = 0.7,
        reg_param: float = 1e-4,
        use_if_weights: bool = False,
        if_sigma: float = 1.0,
        graph_gamma: float = 0.0,
        graph_alpha_in: float = 1.0,
        graph_alpha_p: float = 1.0,
        graph_knn: int = 10,
        graph_threshold: float = 1.0,
        class_sensitive: bool = False,
        random_state: int | None = None,
        use_kernel_for_graph: bool = True,
        graph_strategy: Literal["discriminative", "knn"] = "discriminative",
        if_strategy: Literal["simple", "geib"] = "simple",
        discriminative_beta: float = 0.3,
        if_delta: float = 0.5,
        if_min_weight: float = 1e-4,
    ) -> None:
        self.n_map_trees = n_map_trees
        self.n_enhance_trees = n_enhance_trees
        self.n_increment_layers = n_increment_layers
        self.tree_max_depth = tree_max_depth
        self.tree_min_samples_split = tree_min_samples_split
        self.tree_max_features_ratio = tree_max_features_ratio
        self.reg_param = reg_param
        self.use_if_weights = use_if_weights
        self.if_sigma = if_sigma
        self.graph_gamma = graph_gamma
        self.graph_alpha_in = graph_alpha_in
        self.graph_alpha_p = graph_alpha_p
        self.graph_knn = graph_knn
        self.graph_threshold = graph_threshold
        self.class_sensitive = class_sensitive
        self.random_state = random_state
        self.use_kernel_for_graph = use_kernel_for_graph
        self.graph_strategy = graph_strategy
        self.if_strategy = if_strategy
        self.discriminative_beta = discriminative_beta
        self.if_delta = if_delta
        self.if_min_weight = if_min_weight
        self.fitted_ = False

    def _build_mapping_trees(
        self, X: NDArray[np.float64], y: NDArray[np.int64], rng: np.random.RandomState
    ) -> tuple[list[RegressionTreeModule], NDArray[np.float64]]:
        """Build the mapping trees and return them with their stacked outputs."""
        trees: list[RegressionTreeModule] = []
        z_parts = []
        for _ in range(self.n_map_trees):
            tree = RegressionTreeModule(
                max_depth=self.tree_max_depth,
                min_samples_split=self.tree_min_samples_split,
                max_features_ratio=self.tree_max_features_ratio,
                random_state=self.random_state,
            )
            tree.fit(X, y, rng)
            z_parts.append(tree.predict(X))
            trees.append(tree)
        return trees, np.hstack(z_parts)

    def _build_enhancement_trees(
        self, A: NDArray[np.float64], y: NDArray[np.int64], rng: np.random.RandomState
    ) -> tuple[list[RegressionTreeModule], NDArray[np.float64]]:
        """Build the enhancement trees on top of the mapping outputs."""
        trees: list[RegressionTreeModule] = []
        h_parts = []
        for _ in range(self.n_enhance_trees):
            tree = RegressionTreeModule(
                max_depth=self.tree_max_depth,
                min_samples_split=self.tree_min_samples_split,
                max_features_ratio=self.tree_max_features_ratio,
                random_state=self.random_state,
            )
            tree.fit(A, y, rng)
            h_parts.append(tree.predict(A))
            trees.append(tree)
        return trees, np.hstack(h_parts)

    def _solve_weights(
        self,
        A: NDArray[np.float64],
        Y_onehot: NDArray[np.float64],
        S: NDArray[np.float64] | None,
        L: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Solve weighted ridge regression ``W = (A^T S A + λI + γ A^T L A)^{-1} A^T S Y``."""
        _, d = A.shape
        if S is not None:
            sw = S @ A
            atsa = A.T @ sw
            atsy = A.T @ (S @ Y_onehot)
        else:
            atsa = A.T @ A
            atsy = A.T @ Y_onehot

        reg = self.reg_param * np.eye(d)
        if L is not None and self.graph_gamma > 0:
            a_t_l_a = A.T @ L @ A
            cov = atsa + reg + self.graph_gamma * a_t_l_a
        else:
            cov = atsa + reg

        # Pseudoinverse for stability.
        try:
            w: NDArray[np.float64] = pinv(cov) @ atsy
        except np.linalg.LinAlgError:
            # Fallback to ordinary pseudoinverse of A (no regularization).
            w = pinv(A) @ Y_onehot
        return w

    def _increment_layer(
        self,
        A_old: NDArray[np.float64],
        y_enc: NDArray[np.int64],
        rng: np.random.RandomState,
    ) -> tuple[NDArray[np.float64], list[RegressionTreeModule]]:
        """Add one group of enhancement trees. Returns the new ``A`` and trees."""
        new_trees: list[RegressionTreeModule] = []
        h_new_parts = []
        for _ in range(self.n_enhance_trees):
            tree = RegressionTreeModule(
                max_depth=self.tree_max_depth,
                min_samples_split=self.tree_min_samples_split,
                max_features_ratio=self.tree_max_features_ratio,
                random_state=self.random_state,
            )
            tree.fit(A_old, y_enc, rng)
            h_new_parts.append(tree.predict(A_old))
            new_trees.append(tree)
        h_new = np.hstack(h_new_parts)
        a_new = np.hstack([A_old, h_new])
        return a_new, new_trees

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> TBLS:
        """Fit the TBLS estimator.

        Args:
            X: Training samples of shape ``(n_samples, n_features)``.
            y: Integer target labels of shape ``(n_samples,)``.

        Returns:
            The fitted estimator (self).
        """
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int).ravel()

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_enc: NDArray[np.int64] = self.label_encoder_.fit_transform(y_arr)
        self.classes_: NDArray[np.int64] = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        # Scale input (CCA outputs are already scaled, but this is safe)
        self.scaler_ = StandardScaler()
        x_scaled = self.scaler_.fit_transform(x_arr)

        # Random number generator (RandomState works for both int and None).
        rng = np.random.RandomState(self.random_state)

        # Compute kernel matrix only for the branches that need it: the GEIB
        # IFS formula and the kNN graph (when use_kernel_for_graph). The
        # discriminative graph and the simple IFS formula are label/Euclidean
        # only and do not need a kernel.
        need_kernel = (self.use_if_weights and self.if_strategy == "geib") or (
            self.graph_gamma > 0 and self.graph_strategy == "knn" and self.use_kernel_for_graph
        )
        k_mat = _kernel.compute_kernel_matrix(x_scaled) if need_kernel else None

        # Intuitionistic fuzzy weights.
        if self.use_if_weights:
            if self.if_strategy == "simple":
                s_vec = _ifs.compute_if_scores_simple(
                    x_scaled,
                    y_enc,
                    sigma_if=self.if_sigma,
                    delta_if=self.if_delta,
                    min_weight=self.if_min_weight,
                )
                s_mat = np.diag(s_vec)
            elif self.if_strategy == "geib":
                s_mat = _ifs.compute_if_scores_geib(
                    x_scaled, y_enc, K=k_mat, if_sigma=self.if_sigma
                )
            else:
                raise ValueError(
                    f"Unsupported if_strategy: {self.if_strategy!r}. Expected 'simple' or 'geib'."
                )
        else:
            s_mat = None

        # Graph Laplacian.
        if self.graph_gamma > 0:
            if self.graph_strategy == "discriminative":
                l_mat = _graph.build_discriminative_graph_laplacian(
                    y_enc, discriminative_beta=self.discriminative_beta
                )
            elif self.graph_strategy == "knn":
                l_mat = _graph.build_graph_laplacian(
                    x_scaled,
                    y_enc,
                    K=k_mat,
                    graph_alpha_in=self.graph_alpha_in,
                    graph_alpha_p=self.graph_alpha_p,
                    graph_knn=self.graph_knn,
                    use_kernel=self.use_kernel_for_graph,
                )
            else:
                raise ValueError(
                    f"Unsupported graph_strategy: {self.graph_strategy!r}. "
                    "Expected 'discriminative' or 'knn'."
                )
        else:
            l_mat = None

        # 1. Feature mapping trees
        self.map_trees_, z = self._build_mapping_trees(x_scaled, y_enc, rng)

        # 2. Enhancement trees (input = Z, not original X) to keep dimension low.
        a_enh = z
        self.enh_trees_, h = self._build_enhancement_trees(a_enh, y_enc, rng)
        a = np.hstack([z, h])

        # 3. One-hot target (class-sensitive disabled)
        y_onehot = np.eye(self.n_classes_)[y_enc]

        # 4. Output weights
        self.W_ = self._solve_weights(a, y_onehot, s_mat, l_mat)
        self.A_ = a
        self.S_ = s_mat
        self.L_ = l_mat
        self.Y_onehot_ = y_onehot

        # 5. Incremental layers (SPI)
        self.inc_trees_layers_: list[list[RegressionTreeModule]] = []
        for _ in range(self.n_increment_layers):
            a_new, new_trees = self._increment_layer(self.A_, y_enc, rng)
            # Recompute weights after adding the layer.
            self.W_ = self._solve_weights(a_new, y_onehot, s_mat, l_mat)
            self.inc_trees_layers_.append(new_trees)
            self.A_ = a_new

        self.fitted_ = True
        return self

    def _compute_raw_output(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the raw linear output for ``X``."""
        x_scaled = self.scaler_.transform(X)

        # Mapping trees
        z = np.hstack([tree.predict(x_scaled) for tree in self.map_trees_])

        # Enhancement trees on Z
        a_enh = z
        h = np.hstack([tree.predict(a_enh) for tree in self.enh_trees_])
        a = np.hstack([z, h])

        # Incremental layers
        for layer_trees in self.inc_trees_layers_:
            h_new = np.hstack([tree.predict(a) for tree in layer_trees])
            a = np.hstack([a, h_new])

        return a @ self.W_

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities (softmax of the raw output)."""
        if not self.fitted_:
            raise RuntimeError("Model not fitted.")
        raw = self._compute_raw_output(X)
        exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
        sum_exp = exp_raw.sum(axis=1, keepdims=True)
        prob = exp_raw / np.maximum(sum_exp, 1e-10)
        return prob.astype(np.float32)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict class labels."""
        prob = self.predict_proba(X)
        return self.classes_[np.argmax(prob, axis=1)]
