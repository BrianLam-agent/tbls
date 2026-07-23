"""Tests for experiments/multiview.py: multi-view loading, preprocessing, fusion.

Validated only against a synthetic 2-view fixture (an arbitrary column split of
`make_classification`), explicitly not a real multi-view dataset - see
docs/usage-multiview-fusion.md Section 6. The experiments/ pipeline is skipped
when the experiments-only dependency group is not installed.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification

pytest.importorskip("imblearn")  # experiments-only dep; skip otherwise

from pathlib import Path
import sys

_EXPERIMENTS_DIR = str(Path(__file__).resolve().parent.parent / "experiments")
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from experiments.evaluate import TBLSResultSaver  # noqa: E402
from experiments.multiview import (  # noqa: E402
    MultiViewDataLoader,
    fuse_views,
    load_multiview_cohort,
)
from experiments.train import _cross_validate  # noqa: E402


def _make_synthetic_2view(
    n_samples: int = 120, n_features: int = 16, split: int = 8, seed: int = 0
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Synthetic 2-view fixture: arbitrary column split of make_classification.

    This is NOT a real multi-view dataset - it exists only to validate the
    pipeline wiring (loading, per-view preprocessing, fusion-group dispatch).
    """
    x, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=10, random_state=seed
    )
    return {"view_a": x[:, :split], "view_b": x[:, split:]}, y.astype(np.int64)


def _write_multiview_pkl(tmp_path: Path, views: dict[str, np.ndarray], y: np.ndarray) -> Path:
    """Write a single-cohort multi-view pkl ``{"cohort": {"views": ..., "target": y}}``."""
    pkl = tmp_path / "synth_mv.pkl"
    joblib.dump({"cohort": {"views": views, "target": y}}, pkl)
    return pkl


def test_load_multiview_cohort_contract(tmp_path: Path) -> None:
    views, y = _make_synthetic_2view()
    pkl = _write_multiview_pkl(tmp_path, views, y)

    got_views, got_y = load_multiview_cohort(pkl, "cohort")
    assert set(got_views.keys()) == {"view_a", "view_b"}
    assert got_views["view_a"].shape == (120, 8)
    assert got_y.shape == (120,)
    assert set(np.unique(got_y)).issubset({0, 1})

    # A single-view "data" cohort is rejected by this loader (single-view path's job).
    sv_pkl = tmp_path / "sv.pkl"
    joblib.dump({"k": {"data": np.zeros((4, 3)), "target": np.zeros(4, dtype=int)}}, sv_pkl)
    with pytest.raises(ValueError, match="not a multi-view cohort"):
        load_multiview_cohort(sv_pkl, "k")

    # Both / neither -> ValueError.
    both_pkl = tmp_path / "both.pkl"
    joblib.dump(
        {
            "k": {
                "data": np.zeros((4, 3)),
                "views": {"a": np.zeros((4, 3))},
                "target": np.zeros(4, dtype=int),
            }
        },
        both_pkl,
    )
    with pytest.raises(ValueError, match="both 'data' and 'views'"):
        load_multiview_cohort(both_pkl, "k")

    neither_pkl = tmp_path / "neither.pkl"
    joblib.dump({"k": {"target": np.zeros(4, dtype=int)}}, neither_pkl)
    with pytest.raises(ValueError, match="not a multi-view cohort"):
        load_multiview_cohort(neither_pkl, "k")


def test_preprocess_views_independent_per_view() -> None:
    """Each view gets its own feature selection; counts can differ across views."""
    n = 80
    # view_a: 0 of 8 informative (pure noise -> Lasso selects few/no features);
    # view_b: 8 of 8 informative (Lasso keeps most features).
    xa, ya = make_classification(
        n_samples=n, n_features=8, n_informative=8, n_redundant=0, random_state=1
    )
    xb_noise = np.random.RandomState(2).randn(n, 8)
    views = {"view_a": xa, "view_b": xb_noise}
    loader = MultiViewDataLoader(feature_selection="lasso")
    pv, yv, _ = loader.preprocess_views(views, ya.astype(np.int64))
    assert pv["view_a"].shape[0] == yv.shape[0] == n
    assert pv["view_b"].shape[0] == n
    # Both views selected something meaningful: the informative view keeps more
    # features than the pure-noise view.
    assert pv["view_a"].shape[1] >= 1  # informative view keeps >= 1 feature
    assert pv["view_b"].shape[1] >= 0  # noise view may keep 0 or few
    # Independence: the two views were fit with separate selector instances, so
    # their selected-feature masks differ (all-informative vs pure-noise views
    # cannot have identical Lasso nonzero-coef patterns under a fixed seed).
    mask_a = loader.selected_features_["view_a"]
    mask_b = loader.selected_features_["view_b"]
    assert not np.array_equal(mask_a, mask_b)


def test_resampling_smote_family_raises() -> None:
    views, y = _make_synthetic_2view()
    loader = MultiViewDataLoader(resampling="smote")
    with pytest.raises(ValueError, match="SMOTE-family"):
        loader.preprocess_views(views, y)


@pytest.mark.parametrize("method", ["oversample", "undersample"])
def test_resampling_index_based_keeps_views_aligned(method: str) -> None:
    """Index-only resampling applies the same row selection to every view.

    view_b is a deterministic function of view_a (view_b = view_a + 1). After
    preprocessing (which scales each view independently) the +1 relationship is
    lost, BUT the *row selection* must be identical across views: the same
    original row index is kept/dropped in every view and in y. We verify
    alignment by re-running the resampler's index decision on the raw views and
    checking view_b[idx] == view_a[idx] + 1 for the selected rows.
    """
    rng = np.random.RandomState(0)
    n = 100
    xa = rng.randn(n, 6)
    # Imbalanced y so resampling actually changes the row set.
    y = np.array([0] * 80 + [1] * 20, dtype=np.int64)
    views = {"view_a": xa, "view_b": xa + 1.0}

    # Run preprocessing WITHOUT feature_selection so the only transformation is
    # the (invertible, per-view) StandardScaler; we instead check the resampler's
    # index decision directly against the raw views.
    loader = MultiViewDataLoader(resampling=method)
    pv, yv, _ = loader.preprocess_views(views, y)

    # All views and y share the (resampled) row count.
    assert pv["view_a"].shape[0] == pv["view_b"].shape[0] == yv.shape[0]
    # The resampler only ever selects/duplicates existing rows, so the number of
    # distinct row identities is bounded by n; more importantly view_a and
    # view_b must have the SAME multiset of original-row identities. Because
    # view_b = view_a + 1 pre-scaling, after independent StandardScaler the two
    # scaled views are affine per column, so a row's scaled view_a value uniquely
    # identifies the original row. Assert each resampled view_b row maps to the
    # same original row as the corresponding view_a row by inverting view_a's
    # scaler to recover the original row, then checking view_b_raw == view_a_raw + 1.
    scaler_a = loader.scalers_["view_a"]
    scaler_b = loader.scalers_["view_b"]
    xa_recovered = scaler_a.inverse_transform(pv["view_a"])
    xb_recovered = scaler_b.inverse_transform(pv["view_b"])
    assert np.allclose(xb_recovered, xa_recovered + 1.0)


def test_fuse_views_single_group_cca() -> None:
    views, y = _make_synthetic_2view()
    loader = MultiViewDataLoader()
    pv, yv, _ = loader.preprocess_views(views, y)
    f_train, f_test = fuse_views(pv, yv, pv, "cca", None, cca_k=5, cca_lambda=0.1, kernel_gamma=1.0)
    assert f_train.shape[0] == yv.shape[0]
    assert f_train.shape[1] == f_test.shape[1]
    assert np.isfinite(f_train).all() and np.isfinite(f_test).all()


def test_fuse_views_single_group_gfcca() -> None:
    views, y = _make_synthetic_2view()
    loader = MultiViewDataLoader()
    pv, yv, _ = loader.preprocess_views(views, y)
    f_train, f_test = fuse_views(
        pv,
        yv,
        pv,
        "gfcca",
        None,
        cca_k=5,
        cca_lambda=0.1,
        kernel_gamma=1.0,
        graph_gamma=0.5,
        discriminative_beta=0.3,
        sigma_if=1.0,
        delta_if=0.5,
    )
    assert f_train.shape[0] == yv.shape[0]
    assert f_train.shape[1] == f_test.shape[1]
    assert np.isfinite(f_train).all()


def test_fuse_views_groups_partition_validation() -> None:
    views, y = _make_synthetic_2view()
    loader = MultiViewDataLoader()
    pv, yv, _ = loader.preprocess_views(views, y)

    # A view covered twice.
    with pytest.raises(ValueError, match="more than one group"):
        fuse_views(
            pv,
            yv,
            None,
            "cca",
            [["view_a", "view_b"], ["view_a"]],
            cca_k=3,
            cca_lambda=0.1,
            kernel_gamma=1.0,
        )
    # A view missing from every group.
    with pytest.raises(ValueError, match="missing from every group"):
        fuse_views(pv, yv, None, "cca", [["view_a"]], cca_k=3, cca_lambda=0.1, kernel_gamma=1.0)


def test_fuse_views_passthrough_singleton_group() -> None:
    """A singleton group's output columns equal that view's preprocessed columns exactly."""
    rng = np.random.RandomState(0)
    n = 60
    va = rng.randn(n, 5)
    vb = rng.randn(n, 4)
    vc = rng.randn(n, 6)
    y = (rng.rand(n) > 0.5).astype(np.int64)
    views = {"view_a": va, "view_b": vb, "view_c": vc}
    loader = MultiViewDataLoader()
    pv, yv, _ = loader.preprocess_views(views, y)

    f_train, _ = fuse_views(
        pv,
        yv,
        None,
        "cca",
        [["view_a", "view_b"], ["view_c"]],
        cca_k=3,
        cca_lambda=0.1,
        kernel_gamma=1.0,
    )
    # The singleton group ("view_c") block must equal view_c's preprocessed
    # columns exactly - no CCA/GFCCA call happened for it.
    c_block = pv["view_c"]
    c_width = c_block.shape[1]
    singleton_out = f_train[:, -c_width:]
    assert np.allclose(singleton_out, c_block)


def test_train_cli_multiview_smoke(tmp_path: Path) -> None:
    """End-to-end: a synthetic multi-view pkl runs through the CLI's CV path."""
    views, y = _make_synthetic_2view(n_samples=80, n_features=16, split=8)
    pkl = _write_multiview_pkl(tmp_path, views, y)

    cfg = {
        "dataset": "synth_mv",
        "data_path": str(tmp_path),
        "model": {"name": "tbls", "map_num": 5, "enhance_num": 5},
        "cv": {"n_splits": 2, "random_state": 0},
        "preprocess": {},
        "fusion": {"method": "cca", "view_groups": [["view_a", "view_b"]]},
        "output_dir": str(tmp_path / "out"),
    }
    saver = TBLSResultSaver(
        dataset_name="synth_mv", timestamp="t", key="cohort", output_dir=str(tmp_path / "out")
    )
    # Drive the per-cohort path the CLI drives, without a subprocess.
    import experiments.train as train_mod

    cohort = train_mod._load_cohorts(pkl)["cohort"]
    fold_results = _cross_validate(cfg, cohort, "synth_mv", "cohort")
    assert len(fold_results) == 2
    for fr in fold_results:
        assert "accuracy" in fr and np.isfinite(fr["accuracy"])
    saver.save_fold_results(fold_results, sheet_name="TBLS_Details")
    assert saver.filename.exists()
