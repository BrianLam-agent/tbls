"""Evaluation metrics and result persistence for TBLS experiments.

`TBLSEvaluator` is a thin wrapper over sklearn metrics for binary **and
multiclass** imbalanced classification. `TBLSResultSaver` writes fold results
and summaries to Excel. Both are self-contained (no `tbls` package coupling)
and were extracted verbatim from the legacy root `tbls.py`.

The metrics schema is owned by :mod:`experiments.metrics_schema` (imported
here rather than redefined) so the logging event schema can reference the same
type without an import cycle.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    log_loss,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from metrics_schema import MetricsDict

logger = logging.getLogger(__name__)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsDict:
    """Binary-classification scalar metrics (no probability-based keys).

    Preserves the exact values the original ``calculate_metrics`` produced for
    the binary path -- this is the regression-tested set; new scalar keys
    (``mcc``/``cohen_kappa``) are additive.

    Args:
        y_true: True 0/1 labels, shape ``(n_samples,)``.
        y_pred: Predicted 0/1 labels, shape ``(n_samples,)``.

    Returns:
        Scalar (non-probability) metrics for binary classification.
    """
    metrics: MetricsDict = {}
    tn, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["negative_predictive_value"] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics["balanced_accuracy"] = (metrics["recall"] + metrics["specificity"]) / 2
    metrics["gmean"] = float(np.sqrt(metrics["recall"] * metrics["specificity"]))
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    return metrics


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsDict:
    """Multiclass scalar metrics (no probability-based keys).

    Specificity / NPV / g-mean are macro-averaged via one-vs-rest
    :func:`sklearn.metrics.multilabel_confusion_matrix`; balanced accuracy
    uses :func:`sklearn.metrics.balanced_accuracy_score` (already multiclass
    aware) rather than the binary hand-derivation.

    Args:
        y_true: True labels, shape ``(n_samples,)``.
        y_pred: Predicted labels, shape ``(n_samples,)``.

    Returns:
        Scalar (non-probability) metrics for multiclass classification.
    """
    metrics: MetricsDict = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision_weighted"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    # One-vs-rest per-class specificity / NPV / gmean, macro-averaged.
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    specs: list[float] = []
    npvs: list[float] = []
    gmeans: list[float] = []
    for cm in mcm:
        tn, fp, fn, tp = cm.ravel()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        specs.append(spec)
        npvs.append(npv)
        gmeans.append(float(np.sqrt(rec * spec)))
    metrics["specificity"] = float(np.mean(specs))
    metrics["negative_predictive_value"] = float(np.mean(npvs))
    metrics["gmean"] = float(np.mean(gmeans))
    return metrics


def _binary_probability_metrics(y_true: np.ndarray, y_score: np.ndarray) -> MetricsDict:
    """Binary-only probability-based metrics.

    ``auroc``/``auprc``/``optimal_threshold`` mirror the original binary
    block (Youden's J argmax over the ROC curve); ``log_loss``/``brier_score``
    are new additive keys. Each key degrades to ``None`` (and a warning is
    logged) if the underlying sklearn call raises -- matching the original
    try/except + warning pattern.

    Args:
        y_true: True 0/1 labels, shape ``(n_samples,)``.
        y_score: Positive-class probability, shape ``(n_samples,)`` (the
            ``(n, 2)`` matrix is reduced to its positive column before this
            helper is called).

    Returns:
        Probability-based binary metrics; failing keys are ``None``.
    """
    metrics: MetricsDict = {}
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_score)
        metrics["auprc"] = average_precision_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        metrics["optimal_threshold"] = thresholds[np.argmax(tpr - fpr)]
    except (ValueError, IndexError) as exc:
        logger.warning("Failed to calculate probability-based metrics: %s", exc)
        metrics["auroc"] = None
        metrics["auprc"] = None
        metrics["optimal_threshold"] = None
    try:
        metrics["log_loss"] = log_loss(y_true, y_score, labels=[0, 1])
    except (ValueError, IndexError) as exc:
        logger.warning("Failed to calculate log_loss: %s", exc)
        metrics["log_loss"] = None
    try:
        metrics["brier_score"] = brier_score_loss(y_true, y_score)
    except (ValueError, IndexError) as exc:
        logger.warning("Failed to calculate brier_score: %s", exc)
        metrics["brier_score"] = None
    return metrics


def _multiclass_probability_metrics(
    y_true: np.ndarray, y_score: np.ndarray, n_classes: int
) -> MetricsDict:
    """Multiclass probability-based metrics (OvR macro AUROC only).

    ``auprc``/``optimal_threshold`` are binary-only concepts (a single
    ROC/PR curve) and are deliberately omitted for multiclass rather than
    forcing a meaningless single-curve reduction; ``log_loss``/``brier_score``
    are likewise binary-only per this plan's scope.

    Args:
        y_true: True labels, shape ``(n_samples,)``.
        y_score: Predicted probabilities, shape ``(n_samples, n_classes)``.
        n_classes: Number of classes.

    Returns:
        ``auroc`` (macro one-vs-rest) if computable, else ``None``.
    """
    metrics: MetricsDict = {}
    try:
        if n_classes > 2:
            metrics["auroc"] = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        else:
            metrics["auroc"] = roc_auc_score(y_true, y_score)
    except (ValueError, IndexError) as exc:
        logger.warning("Failed to calculate multiclass AUROC: %s", exc)
        metrics["auroc"] = None
    return metrics


class TBLSEvaluator:
    """Evaluator for binary and multiclass classification with imbalance support.

    ``calculate_metrics`` dispatches on ``len(np.unique(y_true))``: the binary
    path keeps today's exact metric set and values (regression-tested), the
    multiclass path uses macro/weighted averages and one-vs-rest
    specificity/NPV/gmean. ``mcc`` and ``cohen_kappa`` are returned for both;
    ``log_loss``/``brier_score`` are binary-only (when ``y_score`` is given).
    """

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None = None,
        task: str = "classification",
    ) -> MetricsDict:
        """Calculate evaluation metrics for binary or multiclass classification.

        Args:
            y_true: True labels, shape ``(n_samples,)``.
            y_pred: Predicted labels, shape ``(n_samples,)``.
            y_score: Predicted probabilities, shape ``(n_samples,)`` (binary
                positive-class) or ``(n_samples, n_classes)``. When omitted,
                probability-based keys are absent.
            task: Reserved (currently unused; always classification).

        Returns:
            Dictionary of metrics; see :class:`MetricsDict` for the schema.
        """
        _ = task
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            metrics = _binary_metrics(y_true, y_pred)
        else:
            metrics = _multiclass_metrics(y_true, y_pred)

        if y_score is not None:
            y_score = np.asarray(y_score)
            # For binary callers, reduce ``(n, 2)`` to the positive column
            # (matches the original behaviour exactly).
            if n_classes == 2 and y_score.ndim > 1 and y_score.shape[1] > 1:
                y_score = y_score[:, 1]
            if n_classes == 2:
                metrics.update(_binary_probability_metrics(y_true, y_score))
            else:
                metrics.update(_multiclass_probability_metrics(y_true, y_score, n_classes))
        return metrics

    @staticmethod
    def calculate_average_metrics(metrics_list: list[MetricsDict]) -> dict[str, Any]:
        """Average metrics across multiple folds.

        Args:
            metrics_list: List of per-fold metric dicts.

        Returns:
            Dictionary of average metrics (``avg_`` prefix).
        """
        if not metrics_list:
            return {}

        avg_metrics: dict[str, Any] = {}
        for key in metrics_list[0]:
            first_val = metrics_list[0][key]
            if isinstance(first_val, (int, float, np.number)):
                values = [m[key] for m in metrics_list if m[key] is not None]
                avg_metrics[f"avg_{key}"] = float(np.mean(values)) if values else None
            elif isinstance(first_val, (np.ndarray, list)):
                continue  # Skip array-like metrics.
            else:
                avg_metrics[f"avg_{key}"] = first_val  # Preserve non-numeric values.

        return avg_metrics


class TBLSResultSaver:
    """Save TBLS evaluation results to structured folders.

    Args:
        dataset_name: Name of the dataset.
        timestamp: Timestamp string used for folder naming.
        key: Sub-dataset key (used in the filename).
        model_name: Model name (default ``"tbls"``).
        feature_selection: Feature-selection method name (for the filename).
        resampling: Resampling method name (for the filename).
        output_dir: Output directory. If None, a default ``results/`` path is
            used.
    """

    def __init__(
        self,
        dataset_name: str,
        timestamp: str,
        key: str,
        model_name: str = "tbls",
        feature_selection: str | None = None,
        resampling: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        self.feature_selection = feature_selection
        self.resampling = resampling

        if output_dir is None:
            self.output_dir = Path("results") / f"{model_name}_{dataset_name}" / timestamp
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build the filename.
        filename = f"{key}_{model_name}"
        if self.feature_selection:
            filename += f"_FS-{self.feature_selection}"
        if self.resampling:
            filename += f"_RS-{self.resampling}"
        filename += ".xlsx"

        self.filename = self.output_dir / filename

    def save_fold_results(
        self, fold_results: list[dict[str, Any]], sheet_name: str = "Details"
    ) -> None:
        """Save fold-specific results to an Excel sheet.

        Args:
            fold_results: Results to save.
            sheet_name: Sheet name (default ``"Details"``).
        """
        df = pd.DataFrame(fold_results)
        if not self.filename.exists():
            with pd.ExcelWriter(self.filename, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(
                self.filename,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="replace",
            ) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def save_summary(self, summary_data: list[dict[str, Any]], sheet_name: str = "Summary") -> None:
        """Save summary data (with a metadata sheet) to Excel.

        Args:
            summary_data: Summary rows to save.
            sheet_name: Sheet name (default ``"Summary"``).
        """
        # Ensure the parent directory exists (double-check).
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if self.filename.exists() else "w"
        with pd.ExcelWriter(self.filename, engine="openpyxl", mode=mode) as writer:
            # Metadata.
            meta_df = pd.DataFrame(
                {
                    "Feature_Selection": [self.feature_selection],
                    "Resampling_Method": [self.resampling],
                }
            )
            meta_df.to_excel(writer, sheet_name=f"{sheet_name}_Meta", index=False)

            # Main data.
            df = pd.DataFrame(summary_data)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
