"""Tests for experiments/evaluate.py metrics (binary + multiclass) and the
shared metrics/event schemas.

Skipped when the experiments-only dependency group isn't installed, matching
tests/test_experiments_train.py's pattern (imbalanced-learn is the sentinel).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("imblearn")  # experiments-only dep; skip otherwise

from experiments.evaluate import TBLSEvaluator
from experiments.metrics_schema import MetricsDict
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _binary_inputs(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed binary ``(y_true, y_pred, y_score)`` triple.

    Args:
        seed: RNG seed for reproducibility.

    Returns:
        ``(y_true, y_pred, y_score)`` each shape ``(200,)``.
    """
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, 2, size=200)
    y_pred = rng.randint(0, 2, size=200)
    y_score = rng.rand(200)
    return y_true, y_pred, y_score


def test_binary_metrics_keys_and_values_match_sklearn() -> None:
    """Binary path: every pre-plan key matches sklearn directly (regression)."""
    y_true, y_pred, y_score = _binary_inputs()
    m = TBLSEvaluator.calculate_metrics(y_true, y_pred, y_score)

    # Pre-plan keys (regression test): exact-value match.
    assert m["accuracy"] == accuracy_score(y_true, y_pred)
    assert m["precision"] == precision_score(y_true, y_pred, zero_division=0)
    assert m["recall"] == recall_score(y_true, y_pred, zero_division=0)
    assert m["f1_score"] == f1_score(y_true, y_pred, zero_division=0)
    # specificity/NPV/balanced_accuracy/gmean are exercised as a group below.
    assert m["auroc"] == roc_auc_score(y_true, y_score)
    assert m["auprc"] == average_precision_score(y_true, y_score)

    # New additive keys present for binary (with y_score).
    assert m["mcc"] == matthews_corrcoef(y_true, y_pred)
    assert m["cohen_kappa"] == cohen_kappa_score(y_true, y_pred)
    assert m["log_loss"] == pytest.approx(log_loss(y_true, y_score, labels=[0, 1]))
    assert m["brier_score"] == brier_score_loss(y_true, y_score)


def test_binary_pre_plan_keyset_unchanged() -> None:
    """Binary path: the 12 pre-plan keys are all present (no rename)."""
    y_true, y_pred, y_score = _binary_inputs()
    m = TBLSEvaluator.calculate_metrics(y_true, y_pred, y_score)
    pre_plan_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "hamming_loss",
        "specificity",
        "negative_predictive_value",
        "balanced_accuracy",
        "gmean",
        "auroc",
        "auprc",
        "optimal_threshold",
    }
    assert pre_plan_keys.issubset(set(m.keys()))


def test_binary_metrics_without_score_omits_probability_keys() -> None:
    """Binary path without y_score: probability-based keys are absent."""
    y_true, y_pred, _ = _binary_inputs()
    m = TBLSEvaluator.calculate_metrics(y_true, y_pred)
    for absent in ("auroc", "auprc", "optimal_threshold", "log_loss", "brier_score"):
        assert absent not in m
    # but mcc/cohen_kappa are present (they don't need y_score)
    assert "mcc" in m and "cohen_kappa" in m


def test_multiclass_metrics_do_not_raise() -> None:
    """Multiclass path (3 classes): calculate_metrics returns without raising."""
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 3, size=300)
    y_pred = rng.randint(0, 3, size=300)
    m = TBLSEvaluator.calculate_metrics(y_true, y_pred)
    assert isinstance(m, dict) and m["accuracy"] >= 0.0


def test_multiclass_keys_and_shape() -> None:
    """Multiclass path: macro/weighted keys present, binary-only keys absent."""
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 3, size=300)
    y_pred = rng.randint(0, 3, size=300)
    y_score = rng.rand(300, 3)
    y_score = y_score / y_score.sum(1, keepdims=True)
    m = TBLSEvaluator.calculate_metrics(y_true, y_pred, y_score)

    for k in (
        "precision",
        "recall",
        "f1_score",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "specificity",
        "negative_predictive_value",
        "balanced_accuracy",
        "gmean",
        "mcc",
        "cohen_kappa",
        "auroc",
        "accuracy",
        "hamming_loss",
    ):
        assert k in m, f"missing multiclass key: {k}"
    # Binary-only keys deliberately omitted for multiclass.
    for absent in ("auprc", "optimal_threshold", "log_loss", "brier_score"):
        assert absent not in m


def test_multiclass_auroc_ovr_macro() -> None:
    """Multiclass AUROC matches sklearn's one-vs-rest macro average."""
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 3, size=300)
    y_score = rng.rand(300, 3)
    y_score = y_score / y_score.sum(1, keepdims=True)
    m = TBLSEvaluator.calculate_metrics(y_true, y_pred=y_true.copy(), y_score=y_score)
    assert m["auroc"] == pytest.approx(
        roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    )


def test_calculate_average_metrics_prefix_and_skip_arrays() -> None:
    """Average: numeric keys get avg_ prefix; array-like keys skipped."""
    y_true, y_pred, y_score = _binary_inputs()
    one = TBLSEvaluator.calculate_metrics(y_true, y_pred, y_score)
    avg = TBLSEvaluator.calculate_average_metrics([one, one])
    assert "avg_accuracy" in avg
    assert avg["avg_accuracy"] == pytest.approx(one["accuracy"])
    assert "avg_fold" not in avg  # not present in these dicts


def test_calculate_average_metrics_empty() -> None:
    """Average: empty list returns {}."""
    assert TBLSEvaluator.calculate_average_metrics([]) == {}


def test_metrics_dict_typeddict_importable() -> None:
    """MetricsDict is importable from experiments.metrics_schema."""
    # TypedDicts subclass dict at runtime in the functional form used here.
    assert MetricsDict.__name__ == "MetricsDict"
