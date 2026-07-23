"""Multi-objective fitness for the genetic optimizer (experimental).

This module's TBLS-coupled function (:meth:`MultiObjectiveFitness.calculate`)
references attributes that do not exist on the current :class:`tbls.tbls.TBLS`
API (``predict(trees=...)``, ``mapping_trees``, ``tree.selected_features``). It
is shipped for reuse of the standalone GA machinery but is **not** verified to
run end-to-end against ``TBLS``. See ``docs/design.md`` §15.3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score, roc_auc_score  # type: ignore[import-untyped]

from tbls.ensemble.diversity_metrics import diversity_score
from tbls.tbls import TBLS


class MultiObjectiveFitness:
    """Multi-objective fitness: classification performance + feature diversity.

    Args:
        X_val: Validation features.
        y_val: Validation labels.
        alpha: Weight of the performance term (diversity weight is ``1 - alpha``).
    """

    def __init__(
        self,
        X_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        alpha: float = 0.7,
    ) -> None:
        self.X_val = X_val
        self.y_val = y_val
        self.alpha = alpha  # performance weight
        self.beta = 1 - alpha  # diversity weight

    def calculate(self, model: TBLS, selected_trees: list[int]) -> float:
        """Compute the weighted fitness of a tree subset.

        Note:
            Coupled to TBLS internals (``predict(trees=...)``, ``mapping_trees``,
            ``tree.selected_features``) that do not exist on the current
            :class:`tbls.tbls.TBLS`. Not verified to run end-to-end; see
            ``docs/design.md`` §15.3.
        """
        # 1. Classification performance (weighted F1 + AUC).
        y_pred = model.predict(self.X_val, trees=selected_trees)  # type: ignore[call-arg]
        f1 = f1_score(self.y_val, y_pred, average="weighted")
        y_proba = model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_proba)
        performance = 0.5 * f1 + 0.5 * auc

        # 2. Feature diversity (mean Jaccard distance).
        diversity = diversity_score(
            [model.mapping_trees[i].selected_features for i in selected_trees],
            method="jaccard",
        )
        return float(self.alpha * performance + self.beta * diversity)
