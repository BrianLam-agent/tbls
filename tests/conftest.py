"""Shared test fixtures for the tbls package."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification

# experiments/train.py imports its sibling modules script-style
# (`from dataprocess import ...`), so make experiments/ importable as a
# top-level path for tests that import experiments.train.
_EXPERIMENTS_DIR = str(Path(__file__).resolve().parent.parent / "experiments")
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic binary-classification dataset."""
    x, y = make_classification(
        n_samples=120,
        n_features=15,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return x, y.astype(np.int64)


@pytest.fixture
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic 3-class classification dataset."""
    x, y = make_classification(
        n_samples=150,
        n_features=15,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    return x, y.astype(np.int64)
