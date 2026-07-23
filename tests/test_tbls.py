"""TBLS sklearn-compatibility tests, including IFS/graph and incremental paths."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from tbls import TBLS


def _make_tbls(**kwargs: object) -> TBLS:
    return TBLS(n_map_trees=5, n_enhance_trees=5, random_state=42, **kwargs)


def test_tbls_fit_predict_proba(classification_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = classification_data
    model = _make_tbls()
    model.fit(x, y)
    pred = model.predict(x)
    proba = model.predict_proba(x)
    assert pred.shape == (x.shape[0],)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
    assert len(np.unique(pred)) > 1


def test_tbls_multiclass(multiclass_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = multiclass_data
    model = _make_tbls()
    model.fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (x.shape[0], 3)
    assert np.isfinite(proba).all()


def test_tbls_incremental_layer(classification_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = classification_data
    model = _make_tbls(n_increment_layers=1)
    model.fit(x, y)
    assert np.isfinite(model.predict(x)).all()


def test_tbls_discriminative_graph_and_simple_ifs_default(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Default strategy (discriminative graph + simple IFS) fits and predicts sanely.

    With ``use_if_weights=True`` and ``graph_gamma=0.1`` and no explicit
    strategy, TBLS uses the new defaults ported from GraphFuzzyKCCA's tuned
    formulas.
    """
    x, y = classification_data
    model = _make_tbls(use_if_weights=True, graph_gamma=0.1)
    model.fit(x, y)
    proba = model.predict_proba(x)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
    assert len(np.unique(model.predict(x))) > 1


def test_tbls_knn_graph_and_geib_ifs_backward_compat(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """``graph_strategy="knn"`` + ``if_strategy="geib"`` reproduces pre-switch behavior."""
    x, y = classification_data
    model = _make_tbls(
        use_if_weights=True, graph_gamma=0.1, graph_strategy="knn", if_strategy="geib"
    )
    model.fit(x, y)
    proba = model.predict_proba(x)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
    assert len(np.unique(model.predict(x))) > 1


def test_tbls_invalid_strategy_raises(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = classification_data
    with pytest.raises(ValueError, match="graph_strategy"):
        _make_tbls(graph_gamma=0.1, graph_strategy="bogus").fit(x, y)
    with pytest.raises(ValueError, match="if_strategy"):
        _make_tbls(use_if_weights=True, if_strategy="bogus").fit(x, y)


def test_tbls_clone_roundtrip(classification_data: tuple[np.ndarray, np.ndarray]) -> None:
    _ = classification_data
    model = _make_tbls()
    assert clone(model).get_params() == model.get_params()


def test_tbls_cross_val(classification_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = classification_data
    scores = cross_val_score(_make_tbls(), x, y, cv=3)
    assert np.isfinite(scores).all()
