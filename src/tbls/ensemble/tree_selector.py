"""Tree subset selection and weight assignment.

Standalone (no TBLS coupling): :class:`TreeSelector` operates on plain fitness
and diversity score dictionaries.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted  # type: ignore[import-untyped]


class TreeSelector:
    """Selector and weight assigner for tree subsets.

    Args:
        selection_method: ``'top_k'`` or ``'threshold'``.
        weight_method: ``'uniform'``, ``'performance'`` or ``'diversity'``.
    """

    def __init__(
        self,
        selection_method: str = "top_k",
        weight_method: str = "performance",
    ) -> None:
        self.selection_method = selection_method
        self.weight_method = weight_method
        self.selected_indices_: NDArray[np.intp] | None = None
        self.weights_: NDArray[np.float64] | None = None

    def fit(
        self,
        fitness_scores: dict[int, float],
        diversity_scores: dict[int, float] | None = None,
    ) -> TreeSelector:
        """Select tree subsets and assign weights from fitness/diversity scores.

        Args:
            fitness_scores: Mapping ``{tree_index: fitness_score}``.
            diversity_scores: Optional mapping ``{tree_index: diversity_score}``,
                required when ``weight_method == 'diversity'``.

        Returns:
            The fitted selector.
        """
        if not fitness_scores:
            raise ValueError("Fitness scores cannot be empty")

        keys = list(fitness_scores.keys())
        indices = np.array(keys, dtype=np.intp)
        fitness = np.array([fitness_scores[k] for k in keys], dtype=np.float64)

        # Selection.
        if self.selection_method == "top_k":
            k = max(1, int(len(indices) * 0.5))  # default: top 50%
            selected = np.argsort(fitness)[-k:]
        elif self.selection_method == "threshold":
            threshold = np.median(fitness)
            selected = np.where(fitness >= threshold)[0]
        else:
            raise ValueError(f"Unsupported selection method: {self.selection_method}")

        self.selected_indices_ = indices[selected]

        # Weighting.
        if self.weight_method == "uniform":
            weights = np.ones(len(selected), dtype=np.float64)
        elif self.weight_method == "performance":
            weights = fitness[selected]
        elif self.weight_method == "diversity" and diversity_scores is not None:
            assert self.selected_indices_ is not None
            diversity = np.array(
                [diversity_scores[int(i)] for i in self.selected_indices_],
                dtype=np.float64,
            )
            weights = diversity / (diversity.sum() + 1e-8)
        else:
            raise ValueError("Invalid weight method or missing diversity scores")

        # Normalize weights.
        self.weights_ = weights / (weights.sum() + 1e-8)
        return self

    def get_selected_trees(self) -> NDArray[np.intp]:
        """Return the selected tree indices."""
        if self.selected_indices_ is None:
            raise NotFittedError("TreeSelector must be fitted first")
        return self.selected_indices_

    def get_weights(self) -> NDArray[np.float64]:
        """Return the tree weights."""
        if self.weights_ is None:
            raise NotFittedError("TreeSelector must be fitted first")
        return self.weights_

    def validate(self) -> None:
        """Validate the selector's internal state."""
        check_is_fitted(self, ["selected_indices_", "weights_"])
        if self.selected_indices_ is None or self.weights_ is None:
            raise ValueError("Selector not fitted")
        if len(self.selected_indices_) != len(self.weights_):
            raise ValueError("Indices and weights length mismatch")
