"""TBLS training CLI: YAML config with typer command-line overrides.

Run with::

    uv run --group experiments python experiments/train.py
    uv run --group experiments python experiments/train.py --dataset biomedical_larger --n-splits 3

Loads a (possibly multi-key) pkl dataset from ``experiments/datasets/``, runs
k-fold CV (``KFold``, matching the legacy ``main.py``), fits :class:`tbls.TBLS`
per fold, evaluates with :class:`evaluate.TBLSEvaluator`, and writes Excel
results with :class:`evaluate.TBLSResultSaver`.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import KFold
import typer
import yaml

from dataprocess import DataLoader
from evaluate import TBLSEvaluator, TBLSResultSaver
from tbls import TBLS

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


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


def _cross_validate(
    cfg: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    key: str,
    timestamp: str,
) -> list[dict[str, Any]]:
    """Run k-fold CV on one sub-dataset; return per-fold metrics."""
    model_cfg = cfg.get("model", {})
    pre_cfg = cfg.get("preprocess", {})
    cv_cfg = cfg.get("cv", {})

    n_splits = int(cv_cfg.get("n_splits", 5))
    random_state = int(cv_cfg.get("random_state", 42))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(x), start=1):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Per-fold DataLoader so preprocessing is fit on the train split only.
        loader = DataLoader(
            dataset_name=dataset_name,
            feature_selection=pre_cfg.get("feature_selection"),
            resampling=pre_cfg.get("resampling"),
        )
        x_tr, y_tr, x_te = loader.preprocess(x_train, y_train, x_test)

        model = TBLS(
            n_map_trees=int(model_cfg.get("map_num", 10)),
            n_enhance_trees=int(model_cfg.get("enhance_num", 10)),
            reg_param=float(model_cfg.get("reg_param", 1e-4)),
            random_state=random_state,
        )
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


@app.command()
def train(
    config: Path = typer.Option(Path("experiments/configs/default.yaml"), help="YAML config path."),  # noqa: B008
    dataset: str | None = typer.Option(None, help="Override config dataset name."),
    map_num: int | None = typer.Option(None, help="Override mapping node count."),
    n_splits: int | None = typer.Option(None, help="Override CV fold count."),
    output_dir: str | None = typer.Option(None, help="Override output directory."),
) -> None:
    """Run a TBLS training experiment from a YAML config."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(config.read_text())
    if dataset is not None:
        cfg["dataset"] = dataset
    if map_num is not None:
        cfg.setdefault("model", {})["map_num"] = map_num
    if n_splits is not None:
        cfg.setdefault("cv", {})["n_splits"] = n_splits
    if output_dir is not None:
        cfg["output_dir"] = output_dir

    dataset_name = cfg["dataset"]
    data_path = Path(cfg.get("data_path", "experiments/datasets"))
    pkl_path = data_path / f"{dataset_name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Dataset pkl not found: {pkl_path}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = Path(cfg.get("output_dir", "results_dir"))

    subsets = _load_subsets(pkl_path)
    logger.info("dataset=%s keys=%s", dataset_name, list(subsets.keys()))

    for key, (x, y) in subsets.items():
        logger.info("=== %s / %s : X=%s y=%s ===", dataset_name, key, x.shape, y.shape)
        fold_results = _cross_validate(cfg, x, y, dataset_name, key, timestamp)

        result_dir = out_base / f"tbls_{dataset_name}" / key / timestamp
        saver = TBLSResultSaver(
            dataset_name=dataset_name,
            timestamp=timestamp,
            key=key,
            model_name="tbls",
            feature_selection=cfg.get("preprocess", {}).get("feature_selection"),
            resampling=cfg.get("preprocess", {}).get("resampling"),
            output_dir=str(result_dir),
        )
        saver.save_fold_results(fold_results, sheet_name="TBLS_Details")
        avg = TBLSEvaluator.calculate_average_metrics(fold_results)
        saver.save_summary([{"key": key, **avg}], sheet_name="TBLS_Summary")
        logger.info("dataset=%s key=%s avg=%s", dataset_name, key, avg)


if __name__ == "__main__":
    app()
