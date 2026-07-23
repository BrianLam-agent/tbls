"""Evaluation metrics and result persistence for TBLS experiments.

`TBLSEvaluator` is a thin wrapper over sklearn metrics for binary/imbalanced
classification. `TBLSResultSaver` writes fold results and summaries to Excel.
Both are self-contained (no `tbls` package coupling) and were extracted verbatim
from the legacy root `tbls.py`.
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
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class TBLSEvaluator:
    """Evaluator for binary classification with imbalanced-data support."""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None = None,
        task: str = "classification",
    ) -> dict[str, Any]:
        """Calculate evaluation metrics for binary classification.

        Args:
            y_true: True labels (0 or 1), shape ``(n_samples,)``.
            y_pred: Predicted labels (0 or 1), shape ``(n_samples,)``.
            y_score: Predicted probabilities, shape ``(n_samples,)`` or
                ``(n_samples, 2)``.
            task: Reserved (currently unused; always binary classification).

        Returns:
            Dictionary of metrics.
        """
        _ = task
        metrics: dict[str, Any] = {}

        # Ensure 1D.
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        # Basic metrics.
        tn, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
        metrics["hamming_loss"] = hamming_loss(y_true, y_pred)

        # Specificity and NPV.
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["negative_predictive_value"] = tn / (tn + fn) if (tn + fn) > 0 else 0

        # Imbalanced-data metrics.
        metrics["balanced_accuracy"] = (metrics["recall"] + metrics["specificity"]) / 2
        metrics["gmean"] = float(np.sqrt(metrics["recall"] * metrics["specificity"]))

        # Probability-based metrics.
        if y_score is not None:
            try:
                # Use the positive-class probability for binary classification.
                if y_score.ndim > 1 and y_score.shape[1] > 1:
                    y_score = y_score[:, 1]
                metrics["auroc"] = roc_auc_score(y_true, y_score)
                metrics["auprc"] = average_precision_score(y_true, y_score)
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                metrics["optimal_threshold"] = thresholds[np.argmax(tpr - fpr)]
            except (ValueError, IndexError) as exc:
                logger.warning("Failed to calculate probability-based metrics: %s", exc)
                metrics["auroc"] = None
                metrics["auprc"] = None
                metrics["optimal_threshold"] = None

        return metrics

    @staticmethod
    def calculate_average_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
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
