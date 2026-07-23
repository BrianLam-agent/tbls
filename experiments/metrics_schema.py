"""Typed schema for the metrics dict returned by ``TBLSEvaluator``.

This module owns the single ``MetricsDict`` definition shared by
``experiments.evaluate`` (which produces it) and ``experiments.logging_schema``
(which carries it inside ``FoldCompletedEvent``). Keeping it here rather than
in ``evaluate.py`` avoids an import cycle and makes the schema discoverable
to IDEs/mypy at the seam between the two modules.

The dict is ``total=False`` because the probability-based metrics
(``auroc``/``auprc``/``optimal_threshold``/``log_loss``/``brier_score``) are
only present when a model provides ``y_score`` (``predict_proba``), and some
keys are binary- or multiclass-only (see the per-field comments).
"""

from __future__ import annotations

from typing import TypedDict


class MetricsDict(TypedDict, total=False):
    """Metrics returned by :meth:`TBLSEvaluator.calculate_metrics`.

    Attributes:
        accuracy: Overall accuracy (binary and multiclass).
        precision: Macro-averaged precision in the multiclass path; the
            binary precision (``average="binary"``) otherwise.
        recall: Macro-averaged recall in the multiclass path; binary recall
            otherwise.
        f1_score: Macro-averaged F1 in the multiclass path; binary F1
            otherwise.
        precision_weighted: Weighted-average precision (multiclass only).
        recall_weighted: Weighted-average recall (multiclass only).
        f1_weighted: Weighted-average F1 (multiclass only).
        hamming_loss: Fraction of misclassified samples (binary and multiclass).
        specificity: Binary: true-negative rate. Multiclass: macro-averaged
            one-vs-rest specificity.
        negative_predictive_value: Binary NPV; multiclass: macro-averaged
            one-vs-rest NPV.
        balanced_accuracy: Binary: ``(recall + specificity) / 2``.
            Multiclass: ``sklearn.balanced_accuracy_score``.
        gmean: Geometric mean of recall and specificity (binary); macro mean
            of per-class one-vs-rest g-mean (multiclass).
        mcc: Matthews correlation coefficient (binary and multiclass).
        cohen_kappa: Cohen's kappa (binary and multiclass).
        auroc: Area under the ROC curve. Binary (positive-class score) or
            multiclass macro OvR; ``None`` if it cannot be computed.
        auprc: Area under the precision-recall curve (binary only; ``None``
            if it cannot be computed).
        optimal_threshold: Youden's J threshold from the ROC curve
            (binary only; ``None`` if it cannot be computed).
        log_loss: Cross-entropy log loss (binary, with ``y_score``).
        brier_score: Brier score (binary, with ``y_score``).
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    precision_weighted: float  # multiclass only
    recall_weighted: float  # multiclass only
    f1_weighted: float  # multiclass only
    hamming_loss: float
    specificity: float
    negative_predictive_value: float
    balanced_accuracy: float
    gmean: float
    mcc: float
    cohen_kappa: float
    auroc: float | None
    auprc: float | None  # binary only
    optimal_threshold: float | None  # binary only
    log_loss: float | None  # binary only
    brier_score: float | None  # binary only
