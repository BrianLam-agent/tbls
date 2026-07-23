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
from pathlib import Path
import time
from typing import Any

import joblib
from loguru import logger
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
from logging_schema import (
    FoldCompletedEvent,
    GridSummaryEvent,
    RunFinishedEvent,
    RunStartedEvent,
)
from logging_setup import configure_logging
from multiview import MultiViewDataLoader, fuse_views, load_multiview_cohort
from tbls import TBLS, BroadLearningSystem

app = typer.Typer(add_completion=False)

# Legacy YAML config keys -> TBLS constructor parameter names.
_TBLS_KEY_MAP = {"map_num": "n_map_trees", "enhance_num": "n_enhance_trees"}


def _native(value: Any) -> Any:
    """Coerce a value to a JSON-native Python scalar (numpy -> python).

    Args:
        value: A scalar (possibly a :class:`numpy.generic`) or ``None``.

    Returns:
        ``None`` unchanged; numpy scalars via ``.item()``; anything else as-is.
    """
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _native_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return ``metrics`` with every numpy scalar coerced to native Python.

    Args:
        metrics: A per-fold metrics dict (may include numpy scalars).

    Returns:
        A new dict safe to embed in a loguru-bound JSON event.
    """
    return {k: _native(v) for k, v in metrics.items()}


def _native_params(params: dict[str, Any] | None) -> dict[str, object] | None:
    """Return grid-point params with numpy scalars coerced to native Python.

    Args:
        params: A grid-point hyperparameter dict, or ``None`` for non-grid runs.

    Returns:
        ``None`` unchanged; otherwise a new dict safe to embed in a JSON event.
    """
    if params is None:
        return None
    return {k: _native(v) for k, v in params.items()}


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
    grid_idx: int | None = None,
    predictions_npz: Path | None = None,
) -> list[dict[str, Any]]:
    """Run k-fold CV on one cohort; return per-fold metrics.

    Branches on single-view ``(X, y)`` vs multi-view ``(views, y, "multiview")``
    cohorts. For multi-view, every view and ``y`` are split by the same fold
    indices, per-view-preprocessed + row-aligned-resampled via
    :class:`MultiViewDataLoader`, then fused via :func:`fuse_views` before the
    model fit - see ``docs/usage-multiview-fusion.md``.

    When ``predictions_npz`` is given and this is a non-grid run
    (``grid_point is None``), the raw per-fold ``y_true``/``y_pred``/
    ``y_score`` arrays are persisted to that ``.npz`` side-file (keyed by
    ``{key}_fold{N}_*``) so :mod:`experiments.visualize` can render
    ROC/PR/confusion-matrix plots. Grid runs skip the side-file (it would
    explode in size across all swept points).
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
    preds: dict[str, np.ndarray] = {}
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

        # Persist raw per-fold arrays for ROC/PR/confusion-matrix plots.
        # Only for non-grid runs (grid sweeps would explode the side-file size).
        if predictions_npz is not None and grid_point is None:
            preds[f"{key}_fold{fold}_y_true"] = y_test
            preds[f"{key}_fold{fold}_y_pred"] = y_pred
            preds[f"{key}_fold{fold}_y_score"] = np.asarray(y_score)

        # Emit a structured FoldCompletedEvent to stdout + the JSONL sink.
        fold_event: FoldCompletedEvent = {
            "event": "fold_completed",
            "dataset": dataset_name,
            "cohort_key": key,
            "fold": fold,
            "n_splits": n_splits,
            "metrics": _native_metrics({k: v for k, v in metrics.items() if k != "fold"}),
            "grid_idx": grid_idx,
            "grid_params": _native_params(grid_point),
            "predictions_file": (
                predictions_npz.name if predictions_npz is not None and grid_point is None else None
            ),
        }
        logger.bind(**fold_event).info("fold_completed")
        logger.info(
            f"dataset={dataset_name} key={key} fold={fold}/{n_splits} "
            f"acc={metrics.get('accuracy', float('nan')):.4f}"
        )

    if predictions_npz is not None and grid_point is None and preds:
        predictions_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(predictions_npz, **preds)
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
    n_points = len(ParameterGrid(grid_axes))
    for i, point in enumerate(ParameterGrid(grid_axes), start=1):
        fold_results = _cross_validate(
            cfg, cohort, dataset_name, key, grid_point=point, grid_idx=i, predictions_npz=None
        )
        saver.save_fold_results(fold_results, sheet_name=f"Grid_{i:03d}")
        avg = TBLSEvaluator.calculate_average_metrics(fold_results)
        rows.append({"grid_idx": i, "model": model_name, **point, **avg})
        logger.info(
            f"dataset={dataset_name} key={key} grid {i}/{n_points} {point} "
            f"acc={avg.get('avg_accuracy', float('nan')):.4f}"
        )

    rows.sort(key=_rank_key, reverse=True)
    saver.save_summary(rows, sheet_name="GridSummary")
    winner = rows[0]
    winner_params = {k: winner[k] for k in winner if k in grid_axes}
    _winner_metric = _native(winner.get("avg_balanced_accuracy"))
    summary_event: GridSummaryEvent = {
        "event": "grid_summary",
        "dataset": dataset_name,
        "cohort_key": key,
        "winner_params": {k: _native(winner[k]) for k in grid_axes},
        "winner_metric": float(_winner_metric) if _winner_metric is not None else 0.0,
        "n_grid_points": n_points,
    }
    logger.bind(**summary_event).info("grid_summary")
    logger.info(
        f"dataset={dataset_name} key={key} grid winner: {winner_params} "
        f"(avg_balanced_accuracy={winner.get('avg_balanced_accuracy', float('nan')):.4f})"
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

    # Dual-sink logging (human-readable stdout + structured JSONL file) at the
    # run-level dir. The JSONL and the ``.npz`` predictions side-file live
    # under ``run_dir/logs/``; per-key Excel dirs are siblings of ``run_dir``.
    run_dir = out_base / f"{model_name}_{dataset_name}" / timestamp
    configure_logging(run_dir, dataset_name, timestamp)
    run_start = time.perf_counter()
    started_event: RunStartedEvent = {
        "event": "run_started",
        "dataset": dataset_name,
        "model": model_name,
        "fusion_method": cfg.get("fusion", {}).get("method"),
        "grid": grid,
    }
    logger.bind(**started_event).info("run_started")

    cohorts = _load_cohorts(pkl_path)
    logger.info(
        f"dataset={dataset_name} model={model_name} keys={list(cohorts.keys())} grid={grid}"
    )

    for key, cohort in cohorts.items():
        is_mv = len(cohort) == 3 and cohort[2] == "multiview"
        if is_mv:
            views, y, _ = cohort
            shapes = {n: v.shape for n, v in views.items()}
            logger.info(f"=== {dataset_name} / {key} : multiview views={shapes} y={y.shape} ===")
        else:
            x, y = cohort
            logger.info(f"=== {dataset_name} / {key} : X={x.shape} y={y.shape} ===")
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
            pred_npz = run_dir / "logs" / f"{dataset_name}_{timestamp}_{key}_predictions.npz"
            fold_results = _cross_validate(
                cfg,
                dataset_name=dataset_name,
                key=key,
                cohort=cohort,
                grid_idx=None,
                predictions_npz=pred_npz,
            )
            saver.save_fold_results(fold_results, sheet_name=f"{model_name}_Details")
            avg = TBLSEvaluator.calculate_average_metrics(fold_results)
            saver.save_summary([{"key": key, **avg}], sheet_name=f"{model_name}_Summary")
            logger.info(f"dataset={dataset_name} key={key} avg={avg}")

    finished_event: RunFinishedEvent = {
        "event": "run_finished",
        "dataset": dataset_name,
        "duration_seconds": time.perf_counter() - run_start,
    }
    logger.bind(**finished_event).info("run_finished")


if __name__ == "__main__":
    app()
