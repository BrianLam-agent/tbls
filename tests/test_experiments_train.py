"""Tests for experiments/train.py: model selection and grid search.

These import the experiments/ training pipeline, which depends on the
experiments-only dependency group (imbalanced-learn, pandas, typer, ...). The
whole module is skipped when that group is not installed (e.g. CI, which syncs
only `--group dev`).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

pytest.importorskip("imblearn")  # experiments-only dep; skip otherwise

from experiments.evaluate import TBLSResultSaver
from experiments.hyperparams import BLS_DEFAULTS, TBLS_DEFAULTS
from experiments.train import _build_model, _run_grid

from tbls import TBLS, BroadLearningSystem


def test_build_model_tbls_defaults() -> None:
    model = _build_model({"name": "tbls"})
    assert isinstance(model, TBLS)
    assert model.n_map_trees == TBLS_DEFAULTS["n_map_trees"]
    assert model.n_enhance_trees == TBLS_DEFAULTS["n_enhance_trees"]
    assert model.reg_param == TBLS_DEFAULTS["reg_param"]


def test_build_model_bls_defaults() -> None:
    model = _build_model({"name": "bls"})
    assert isinstance(model, BroadLearningSystem)
    assert model.n_feature_groups == BLS_DEFAULTS["n_feature_groups"]
    assert model.reg_param == BLS_DEFAULTS["reg_param"]


def test_build_model_legacy_keys_override() -> None:
    """Legacy map_num/enhance_num config keys map to n_map_trees/n_enhance_trees."""
    model = _build_model({"name": "tbls", "map_num": 7, "enhance_num": 9, "reg_param": 0.25})
    assert isinstance(model, TBLS)
    assert model.n_map_trees == 7
    assert model.n_enhance_trees == 9
    assert model.reg_param == 0.25


def test_build_model_grid_point_wins() -> None:
    model = _build_model({"name": "tbls", "map_num": 7}, grid_point={"n_map_trees": 13})
    assert isinstance(model, TBLS)
    assert model.n_map_trees == 13  # grid point overrides config


def test_build_model_invalid_name_raises() -> None:
    with pytest.raises(ValueError, match=r"model\.name"):
        _build_model({"name": "bogus"})


def test_grid_smoke_ranks_and_one_row_per_point(tmp_path, monkeypatch) -> None:
    """--grid path: a tiny 2x2 grid on synthetic data yields ranked rows."""
    monkeypatch.setattr(
        "experiments.train.TBLS_GRID", {"n_map_trees": [3, 5], "reg_param": [1e-8, 1e-4]}
    )
    x, y = make_classification(n_samples=60, n_features=8, random_state=0)
    y = y.astype(np.int64)
    cfg = {"model": {"name": "tbls"}, "cv": {"n_splits": 2, "random_state": 0}, "preprocess": {}}
    saver = TBLSResultSaver(dataset_name="synth", timestamp="t", key="k", output_dir=str(tmp_path))

    rows = _run_grid(cfg, x, y, "synth", "k", "tbls", saver)

    assert len(rows) == 4  # 2x2 grid
    assert {r["grid_idx"] for r in rows} == {1, 2, 3, 4}
    accs = [r["avg_balanced_accuracy"] for r in rows]
    assert accs == sorted(accs, reverse=True)  # ranked descending
    for row in rows:
        assert "n_map_trees" in row and "reg_param" in row  # config carried
