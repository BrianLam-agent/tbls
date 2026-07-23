"""BroadLearningSystem sklearn-compatibility tests."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from tbls import BroadLearningSystem


def _make_bls() -> BroadLearningSystem:
    return BroadLearningSystem(
        n_feature_groups=3,
        n_feature_nodes_per_group=15,
        n_enhancement_groups=3,
        n_enhancement_nodes_per_group=15,
        random_state=42,
    )


def test_bls_fit_predict_proba(classification_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = classification_data
    model = _make_bls()
    model.fit(x, y)
    pred = model.predict(x)
    proba = model.predict_proba(x)
    assert pred.shape == (x.shape[0],)
    assert proba.shape == (x.shape[0], model.n_classes_)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
    assert len(np.unique(pred)) > 1


def test_bls_clone_roundtrip(classification_data: tuple[np.ndarray, np.ndarray]) -> None:
    _ = classification_data
    model = _make_bls()
    assert clone(model).get_params() == model.get_params()


def test_bls_cross_val(classification_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = classification_data
    scores = cross_val_score(_make_bls(), x, y, cv=3)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
