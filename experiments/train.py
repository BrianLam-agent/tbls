"""TBLS/BLS training CLI: YAML config with typer command-line overrides.

Run with::

    uv run --group experiments python experiments/train.py
    uv run --group experiments python experiments/train.py --dataset biomedical_larger --n-splits 3
    uv run --group experiments python experiments/train.py --model bls
    uv run --group experiments python experiments/train.py --grid

Loads a (possibly multi-key) pkl dataset from ``experiments/datasets/``, runs
k-fold CV (``KFold``), fits either :class:`tbls.TBLS` or
:class:`tbls.BroadLearningSystem` (selected by ``model.name``), evaluates with
:class:`evaluate.TBLSEvaluator`, and writes Excel results with
:class:`evaluate.TBLSResultSaver`. ``--grid`` sweeps the hyperparameter grid
defined in :mod:`experiments.hyperparams` and writes a ranked ``GridSummary``.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, ParameterGrid
import typer
import yaml

from dataprocess import DataLoader
from evaluate import TBLSEvaluator, TBLSResultSaver
from hyperparams import (
    BLS_DEFAULTS,
    BLS_GRID,
    CCA_DEFAULTS,
    GFCCA_DEFAULTS,
    TBLS_DEFAULTS,
    TBLS_GRID,
)
from multiview import MultiViewDataLoader, fuse_views, load_multiview_cohort
from tbls import TBLS, BroadLearningSystem

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)

# Legacy YAML config keys -> TBLS constructor parameter names.
_TBLS_KEY_MAP = {"map_num": "n_map_trees", "enhance_num": "n_enhance_trees"}

# A cohort is either single-view (np.ndarray X, np.ndarray y) or multi-view
# (dict[str, np.ndarray] views, np.ndarray y); tagged so the per-fold body can
# branch without re-inspecting the raw pkl.
CohortData = tuple[np.ndarray, np.ndarray] | tuple[dict[str, np.ndarray], np.ndarray, str]


def _load_subsets(pkl_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load a pkl and return ``{key: (X, y)}`` for every sub-dataset.

    Handles both a flat ``{'data':..., 'target':...}`` dict (reported under the
    key ``"single"``) and a multi-key dict of sub-datasets. Feature matrices
    with ``dtype=object`` are coerced to ``float64``; label ``-1`` is dropped;
    labels are binarized to ``{0, 1}``.
    """
    data = joblib.load(pkl_path)
    subsets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if isinstance(data, dict) and "data" in data and "target" in data:
        items = {"single": data}
    elif isinstance(data, dict):
        items = {
            k: v for k, v in data.items() if isinstance(v, dict) and "data" in v and "target" in v
        }
    else:
        raise TypeError(f"Unsupported pkl top-level type: {type(data)}")

    for key, sub in items.items():
        x = np.asarray(sub["data"], dtype=np.float64)
        y = np.asarray(sub["target"]).ravel()
        valid = y != -1
        x, y = x[valid], y[valid]
        y = (y > 0).astype(np.int64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        subsets[key] = (x, y)
    return subsets


def _load_cohorts(pkl_path: Path) -> dict[str, CohortData]:
    """Load every cohort, tagged single-view or multi-view by pkl content.

    Each cohort key resolves to exactly one of:
    - single-view: ``(X, y)`` where ``X`` is a ``(n, f)`` matrix;
    - multi-view:  ``(views, y, "multiview")`` where ``views`` is a
      ``{name: (n, f_name)}`` dict.

    A cohort dict with both/neither of ``"data"``/``"views"`` raises
    ``ValueError`` (via :func:`load_multiview_cohort` for the multi-view side;
    single-view keys keep the existing ``"data"`` contract).
    """
    data = joblib.load(pkl_path)
    if isinstance(data, dict) and "data" in data and "target" in data:
        items: dict[str, object] = {"single": data}
    elif isinstance(data, dict):
        items = {
            k: v
            for k, v in data.items()
            if isinstance(v, dict) and ("data" in v or "views" in v) and "target" in v
        }
    else:
        raise TypeError(f"Unsupported pkl top-level type: {type(data)}")

    cohorts: dict[str, CohortData] = {}
    for key, sub in items.items():
        if "views" in sub:
            views, y = load_multiview_cohort(pkl_path, key)
            cohorts[key] = (views, y, "multiview")
        else:
            x = np.asarray(sub["data"], dtype=np.float64)
            y = np.asarray(sub["target"]).ravel()
            valid = y != -1
            x, y = x[valid], y[valid]
            y = (y > 0).astype(np.int64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            cohorts[key] = (x, y)
    return cohorts


def _fusion_kwargs(cfg: dict[str, Any], method: str) -> dict[str, Any]:
    """Build fusion kwargs: ``CCA_DEFAULTS``/``GFCCA_DEFAULTS`` + ``fusion`` overrides."""
    defaults = GFCCA_DEFAULTS if method == "gfcca" else CCA_DEFAULTS
    fusion_cfg = dict(cfg.get("fusion", {}))
    fusion_cfg.pop("method", None)
    fusion_cfg.pop("view_groups", None)
    # Only keep kwargs the build fn actually accepts.
    valid = set(defaults)
    return {**defaults, **{k: v for k, v in fusion_cfg.items() if k in valid}}


def _build_model(
    model_cfg: dict[str, Any],
    grid_point: dict[str, Any] | None = None,
) -> BaseEstimator:
    """Build a ``TBLS`` or ``BroadLearningSystem`` from the model config.

    Defaults come from ``experiments.hyperparams`` (``TBLS_DEFAULTS`` /
    ``BLS_DEFAULTS``); the YAML ``model`` section overrides them, with legacy
    keys ``map_num``/``enhance_num`` translated to ``n_map_trees``/
    ``n_enhance_trees``. When ``grid_point`` is given (a dict of direct
    constructor parameter names), it is applied last and wins.
    """
    name = model_cfg.get("name", "tbls")
    if name == "tbls":
        defaults = TBLS_DEFAULTS
        cls: type[BaseEstimator] = TBLS
    elif name == "bls":
        defaults = BLS_DEFAULTS
        cls = BroadLearningSystem
    else:
        raise ValueError(f"Unsupported model.name: {name!r}. Expected 'tbls' or 'bls'.")

    valid = set(inspect.signature(cls).parameters)
    overrides: dict[str, Any] = {}
    for key, value in model_cfg.items():
        if key == "name":
            continue
        pk = _TBLS_KEY_MAP.get(key, key)
        if pk in valid:
            overrides[pk] = value
    if grid_point:
        for key, value in grid_point.items():
            if key in valid:
                overrides[key] = value
    return cls(**{**defaults, **overrides})


def _cross_validate(
    cfg: dict[str, Any],
    cohort: CohortData,
    dataset_name: str,
    key: str,
    grid_point: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run k-fold CV on one cohort; return per-fold metrics.

    Branches on single-view ``(X, y)`` vs multi-view ``(views, y, "multiview")``
    cohorts. For multi-view, every view and ``y`` are split by the same fold
    indices, per-view-preprocessed + row-aligned-resampled via
    :class:`MultiViewDataLoader`, then fused via :func:`fuse_views` before the
    model fit - see ``docs/usage-multiview-fusion.md``.
    """
    model_cfg = {
        **cfg.get("model", {}),
        "random_state": int(cfg.get("cv", {}).get("random_state", 42)),
    }
    pre_cfg = cfg.get("preprocess", {})
    cv_cfg = cfg.get("cv", {})
    fusion_cfg = cfg.get("fusion", {})

    n_splits = int(cv_cfg.get("n_splits", 5))
    random_state = int(cv_cfg.get("random_state", 42))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    is_multiview = len(cohort) == 3 and cohort[2] == "multiview"
    if is_multiview:
        views: dict[str, np.ndarray] = cohort[0]
        y: np.ndarray = cohort[1]
        split_array = next(iter(views.values()))
    else:
        x: np.ndarray = cohort[0]
        y = cohort[1]
        split_array = x
    method = fusion_cfg.get("method", "gfcca")
    view_groups = fusion_cfg.get("view_groups")
    fkwargs = _fusion_kwargs(cfg, method)

    fold_results: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(split_array), start=1):
        y_train, y_test = y[train_idx], y[test_idx]

        if is_multiview:
            mv_train = {name: xv[train_idx] for name, xv in views.items()}
            mv_test = {name: xv[test_idx] for name, xv in views.items()}
            mv_loader = MultiViewDataLoader(
                feature_selection=pre_cfg.get("feature_selection"),
                resampling=pre_cfg.get("resampling"),
                fusion_reference_view=pre_cfg.get("fusion_reference_view"),
            )
            pv_train, y_tr, pv_test = mv_loader.preprocess_views(mv_train, y_train, mv_test)
            x_tr, x_te = fuse_views(pv_train, y_tr, pv_test, method, view_groups, **fkwargs)
        else:
            # Per-fold DataLoader so preprocessing is fit on the train split only.
            loader = DataLoader(
                dataset_name=dataset_name,
                feature_selection=pre_cfg.get("feature_selection"),
                resampling=pre_cfg.get("resampling"),
            )
            x_tr, y_tr, x_te = loader.preprocess(x[train_idx], y_train, x[test_idx])

        model = _build_model(model_cfg, grid_point=grid_point)
        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_te).ravel()
        y_score = model.predict_proba(x_te)
        metrics = TBLSEvaluator.calculate_metrics(y_test, y_pred, y_score)
        metrics["fold"] = fold
        fold_results.append(metrics)
        logger.info(
            "dataset=%s key=%s fold=%d/%d acc=%.4f",
            dataset_name,
            key,
            fold,
            n_splits,
            metrics.get("accuracy", float("nan")),
        )
    return fold_results


def _rank_key(row: dict[str, Any]) -> float:
    """Sort key for grid rows: avg_balanced_accuracy, missing/None last."""
    value = row.get("avg_balanced_accuracy")
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def _run_grid(
    cfg: dict[str, Any],
    cohort: CohortData,
    dataset_name: str,
    key: str,
    model_name: str,
    saver: TBLSResultSaver,
) -> list[dict[str, Any]]:
    """Sweep the model hyperparameter grid for one cohort; write a ranked summary.

    Scope note: sweeps only the model grid (``TBLS_GRID``/``BLS_GRID``) at a
    fixed fusion default for multi-view cohorts; fusion hyperparameters are not
    swept in this pass.
    """
    name = cfg.get("model", {}).get("name", "tbls")
    grid_axes = TBLS_GRID if name == "tbls" else BLS_GRID

    rows: list[dict[str, Any]] = []
    for i, point in enumerate(ParameterGrid(grid_axes), start=1):
        fold_results = _cross_validate(cfg, cohort, dataset_name, key, grid_point=point)
        saver.save_fold_results(fold_results, sheet_name=f"Grid_{i:03d}")
        avg = TBLSEvaluator.calculate_average_metrics(fold_results)
        rows.append({"grid_idx": i, "model": model_name, **point, **avg})
        logger.info(
            "dataset=%s key=%s grid %d/%d %s acc=%.4f",
            dataset_name,
            key,
            i,
            len(ParameterGrid(grid_axes)),
            point,
            avg.get("avg_accuracy", float("nan")),
        )

    rows.sort(key=_rank_key, reverse=True)
    saver.save_summary(rows, sheet_name="GridSummary")
    winner = rows[0]
    logger.info(
        "dataset=%s key=%s grid winner: %s (avg_balanced_accuracy=%.4f)",
        dataset_name,
        key,
        {k: winner[k] for k in winner if k in grid_axes},
        winner.get("avg_balanced_accuracy", float("nan")),
    )
    return rows


@app.command()
def train(
    config: Path = typer.Option(Path("experiments/configs/default.yaml"), help="YAML config path."),  # noqa: B008
    dataset: str | None = typer.Option(None, help="Override config dataset name."),
    model: str | None = typer.Option(None, help="Override model name (tbls|bls)."),
    map_num: int | None = typer.Option(None, help="Override mapping node count (TBLS)."),
    n_splits: int | None = typer.Option(None, help="Override CV fold count."),
    output_dir: str | None = typer.Option(None, help="Override output directory."),
    fusion: str | None = typer.Option(
        None,
        "--fusion",
        help="Override fusion method for multi-view cohorts (cca|gfcca). "
        "Only overrides *which* fusion runs; fusion always runs for a multi-view cohort.",
    ),
    grid: bool = typer.Option(
        False, "--grid", help="Sweep the hyperparameter grid and write a ranked GridSummary."
    ),
) -> None:
    """Run a TBLS/BLS training experiment from a YAML config."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(config.read_text())
    if dataset is not None:
        cfg["dataset"] = dataset
    if model is not None:
        cfg.setdefault("model", {})["name"] = model
    if map_num is not None:
        cfg.setdefault("model", {})["map_num"] = map_num
    if n_splits is not None:
        cfg.setdefault("cv", {})["n_splits"] = n_splits
    if output_dir is not None:
        cfg["output_dir"] = output_dir
    if fusion is not None:
        cfg.setdefault("fusion", {})["method"] = fusion

    dataset_name = cfg["dataset"]
    data_path = Path(cfg.get("data_path", "experiments/datasets"))
    pkl_path = data_path / f"{dataset_name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Dataset pkl not found: {pkl_path}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = Path(cfg.get("output_dir", "results_dir"))
    model_name = cfg.get("model", {}).get("name", "tbls")

    cohorts = _load_cohorts(pkl_path)
    logger.info(
        "dataset=%s model=%s keys=%s grid=%s",
        dataset_name,
        model_name,
        list(cohorts.keys()),
        grid,
    )

    for key, cohort in cohorts.items():
        is_mv = len(cohort) == 3 and cohort[2] == "multiview"
        if is_mv:
            views, y, _ = cohort
            shapes = {n: v.shape for n, v in views.items()}
            logger.info(
                "=== %s / %s : multiview views=%s y=%s ===", dataset_name, key, shapes, y.shape
            )
        else:
            x, y = cohort
            logger.info("=== %s / %s : X=%s y=%s ===", dataset_name, key, x.shape, y.shape)
        result_dir = out_base / f"{model_name}_{dataset_name}" / key / timestamp
        saver = TBLSResultSaver(
            dataset_name=dataset_name,
            timestamp=timestamp,
            key=key,
            model_name=model_name,
            feature_selection=cfg.get("preprocess", {}).get("feature_selection"),
            resampling=cfg.get("preprocess", {}).get("resampling"),
            output_dir=str(result_dir),
        )

        if grid:
            _run_grid(cfg, cohort, dataset_name, key, model_name, saver)
        else:
            fold_results = _cross_validate(cfg, cohort, dataset_name, key)
            saver.save_fold_results(fold_results, sheet_name=f"{model_name}_Details")
            avg = TBLSEvaluator.calculate_average_metrics(fold_results)
            saver.save_summary([{"key": key, **avg}], sheet_name=f"{model_name}_Summary")
            logger.info("dataset=%s key=%s avg=%s", dataset_name, key, avg)


if __name__ == "__main__":
    app()
