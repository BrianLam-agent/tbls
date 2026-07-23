"""Visualization CLI for TBLS experiment runs.

Reads one or more experiment run directories (each containing a ``logs/*.jsonl``
structured log produced by :mod:`experiments.train` via
:mod:`experiments.logging_setup`) and renders static matplotlib plots:

- **per-fold metric bars** (``balanced_accuracy`` / ``mcc`` per cohort) from
  the scalar ``FoldCompletedEvent`` metrics -- always in scope;
- **grid-search summary** (a metric vs. each swept hyperparameter, one
  subplot per swept param) from the grid-point ``FoldCompletedEvent`` rows;
- **ROC / PR curves** and a **confusion-matrix heatmap** from the raw per-fold
  ``y_true`` / ``y_score`` / ``y_pred`` arrays persisted to the
  ``logs/*_predictions.npz`` side-file (non-grid runs only; grid runs skip [src]
  these raw-array plots since the side-file is not produced for them -- see
  ``docs/usage-experiments-cli.md``).

When multiple ``--dir`` paths are given, each run is tagged (by its dataset /
model / run timestamp parsed from the JSONL ``run_started`` event) and the
runs are overlaid/faceted on the same figures so ablation variants or
grid-search runs can be compared at a glance.

Run with::

    uv run --group experiments python experiments/visualize.py --dir <run1> --dir <run2> --output-dir plots/comparison

The output PNGs are written to ``--output-dir`` (default: ``plots/`` next to
the first ``--dir``), one file per plot type, deterministically named.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive; write PNGs only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)


def _repo_root() -> Path:
    """Return the repo root (parent of this file's ``experiments/`` dir).

    Needed so the script can be run from anywhere and still resolve the
    ``experiments/`` sibling imports used only for schema constants.
    """
    return Path(__file__).resolve().parent.parent


# Make ``experiments/`` sibling imports resolvable (mirrors train.py's path
# setup, which the CLI entrypoint already does at import time via its own
# sys.path manipulation in the experiments package). For visualize.py this is
# only needed if it is imported as a module from the repo root.
_REPO_ROOT = _repo_root()
for _p in (_REPO_ROOT, _REPO_ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _run_label(run_started: dict[str, Any] | None, run_dir: Path) -> str:
    """Build a short, stable run label from the ``run_started`` event.

    Args:
        run_started: The parsed ``run_started`` event dict, or ``None``.
        run_dir: The run directory (used as a fallback label).

    Returns:
        A label like ``tbls_biomedical_larger/20260724_034914``.
    """
    if run_started is not None:
        ds = run_started.get("dataset", "?")
        mdl = run_started.get("model", "?")
        return f"{mdl}_{ds}/{run_dir.name}"
    return run_dir.name


def collect_events(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, str]:
    """Parse every ``logs/*.jsonl`` line under ``run_dir`` into event rows.

    Args:
        run_dir: A run directory containing ``logs/*.jsonl``.

    Returns:
        ``(fold_rows, grid_summary_rows, run_started_or_None, run_label)``.
        ``fold_rows`` are ``FoldCompletedEvent``s (``grid_params`` exploded into
        ``param_<name>`` columns); ``grid_summary_rows`` are
        ``GridSummaryEvent``s verbatim.
    """
    fold_rows: list[dict[str, Any]] = []
    grid_summary_rows: list[dict[str, Any]] = []
    run_started: dict[str, Any] | None = None
    for jl in sorted(run_dir.glob("**/logs/*.jsonl")):
        with jl.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                extra: dict[str, Any] = rec.get("record", {}).get("extra", {}) or {}
                event = extra.get("event")
                if event == "run_started":
                    if run_started is None:
                        run_started = extra
                elif event == "fold_completed":
                    row = {k: v for k, v in extra.items() if k != "metrics"}
                    row.update(extra.get("metrics", {}))
                    gp = extra.get("grid_params") or {}
                    for k, v in gp.items():
                        row[f"param_{k}"] = v
                    fold_rows.append(row)
                elif event == "grid_summary":
                    grid_summary_rows.append(extra)
    label = _run_label(run_started, run_dir)
    return fold_rows, grid_summary_rows, run_started, label


def _metric_columns(df: pd.DataFrame) -> list[str]:
    """Return the numeric metric columns present in the fold DataFrame.

    Args:
        df: A fold DataFrame built from ``collect_events``.

    Returns:
        Sorted metric-column names (excluding bookkeeping / grid-param columns).
    """
    skip = {
        "event",
        "dataset",
        "cohort_key",
        "fold",
        "n_splits",
        "grid_idx",
        "grid_params",
        "predictions_file",
    }
    out: list[str] = []
    for c in df.columns:
        if c in skip or c.startswith("param_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return sorted(out)


def plot_per_fold_bars(fold_df: pd.DataFrame, out_dir: Path) -> Path:
    """Bar plot of ``balanced_accuracy`` and ``mcc`` per cohort, grouped by run.

    Args:
        fold_df: Fold rows across all runs (must include a ``run`` tag column).
        out_dir: Directory to write the PNG into (created if missing).

    Returns:
        Path to the written PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["balanced_accuracy", "mcc"], strict=False):
        if metric not in fold_df.columns:
            ax.set_title(f"{metric} (absent)")
            ax.set_axis_off()
            continue
        pivot = fold_df.pivot_table(
            index="cohort_key", columns="run", values=metric, aggfunc="mean"
        )
        pivot.plot(kind="bar", ax=ax, capsize=3)
        ax.set_title(f"Mean {metric} per cohort")
        ax.set_ylabel(metric)
        ax.set_xlabel("cohort")
        ax.set_ylim(bottom=max(0, float(pivot.min().min()) - 0.1))
        ax.legend(title="run")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Per-fold metrics across cohorts (mean over folds)")
    fig.tight_layout()
    path = out_dir / "per_fold_metrics.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_grid_summary(fold_df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Plot the primary metric vs. each swept hyperparameter (one subplot each).

    Args:
        fold_df: Fold rows across all runs (grid rows carry ``param_<name>``).
        out_dir: Directory to write the PNG into (created if missing).

    Returns:
        Path to the written PNG, or ``None`` if no grid rows were present.
    """
    grid_df = fold_df[fold_df["grid_idx"].notna()].copy()
    if grid_df.empty:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    swept = [c[len("param_") :] for c in grid_df.columns if c.startswith("param_")]
    if not swept:
        return None
    metric = "balanced_accuracy"
    if metric not in grid_df.columns:
        metric = next((c for c in _metric_columns(grid_df) if c == "balanced_accuracy"), None)  # type: ignore[arg-type]
        if metric is None:
            return None
    # Average the metric over folds (and over the other swept axes) per
    # (run, cohort_key, swept_axis value).
    n = len(swept)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, param in zip(axes[0], swept, strict=False):
        col = f"param_{param}"
        agg = grid_df.groupby(["run", "cohort_key", col], dropna=False)[metric].mean().reset_index()
        for (run, cohort), sub in agg.groupby(["run", "cohort_key"]):
            sub = sub.sort_values(col)
            ax.plot(sub[col], sub[metric], marker="o", label=f"{cohort} ({run})")
        ax.set_xlabel(param)
        ax.set_ylabel(f"mean {metric}")
        ax.set_title(f"{metric} vs {param}")
        if col == "param_reg_param":
            ax.set_xscale("log")
        ax.legend(fontsize=7)
    fig.suptitle("Grid-search: metric vs swept hyperparameter")
    fig.tight_layout()
    path = out_dir / "grid_search_summary.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _load_predictions(logs_dir: Path, predictions_file: str) -> dict[str, np.ndarray]:
    """Load the ``.npz`` side-file for ``predictions_file`` under ``logs_dir``.

    Args:
        logs_dir: The specific ``logs/`` directory that holds the side-file --
            i.e. the parent directory of the ``.jsonl`` file the event was read
            from, NOT necessarily the top-level ``--dir`` passed on the CLI
            (which may be a shallower ancestor when ``--dir`` spans multiple
            timestamped runs via the recursive ``**/logs/*.jsonl`` discovery).
        predictions_file: The side-file basename recorded in the event.

    Returns:
        A ``{key: array}`` mapping (e.g. ``{"DM_fold1_y_true": array, ...}``).
    """
    npz_path = logs_dir / predictions_file
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def _cohort_roc(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Compute a binary ROC curve (FPR, TPR, AUC) for one cohort fold.

    Args:
        y_true: True 0/1 labels.
        y_score: Positive-class probability (the ``(n, 2)`` matrix is reduced to
            its positive column).

    Returns:
        ``(fpr, tpr, auc)`` or ``None`` if ROC cannot be computed (single
        class, degenerate scores, etc.).
    """
    from sklearn.metrics import auc, roc_curve

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score)
    if y_score.ndim > 1 and y_score.shape[1] > 1:
        y_score = y_score[:, 1]
    if len(np.unique(y_true)) < 2:
        return None
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return fpr, tpr, float(auc(fpr, tpr))
    except ValueError:
        return None


def _cohort_pr(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Compute a binary PR curve (recall, precision, AP) for one cohort fold.

    Args:
        y_true: True 0/1 labels.
        y_score: Positive-class probability.

    Returns:
        ``(recall, precision, ap)`` or ``None`` if PR cannot be computed.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score)
    if y_score.ndim > 1 and y_score.shape[1] > 1:
        y_score = y_score[:, 1]
    if len(np.unique(y_true)) < 2:
        return None
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = float(average_precision_score(y_true, y_score))
        return recall, precision, ap
    except ValueError:
        return None


def _cohort_predictions(
    run_dir: Path, run_tag: str
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Aggregate raw per-fold predictions per cohort from a run's npz side-files.

    Args:
        run_dir: The run directory.
        run_tag: The run label (used only for logging skipped files).

    Returns:
        ``{(run_tag, cohort_key): {"y_true": [...], "y_pred": [...],
        "y_score": [...]}}`` with each fold's arrays concatenated.
    """
    by_cohort: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    for jl in sorted(run_dir.glob("**/logs/*.jsonl")):
        with jl.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                extra = rec.get("record", {}).get("extra", {}) or {}
                if extra.get("event") != "fold_completed":
                    continue
                pf = extra.get("predictions_file")
                if not pf:
                    continue
                cohort = extra.get("cohort_key", "?")
                fold = int(extra["fold"])
                try:
                    arrays = _load_predictions(jl.parent, pf)
                except FileNotFoundError:
                    continue
                key = run_tag, cohort
                slot = by_cohort.setdefault(key, {"y_true": [], "y_pred": [], "y_score": []})
                slot["y_true"].append(arrays[f"{cohort}_fold{fold}_y_true"])
                slot["y_pred"].append(arrays[f"{cohort}_fold{fold}_y_pred"])
                slot["y_score"].append(arrays[f"{cohort}_fold{fold}_y_score"])
    return {k: {a: np.concatenate(v) for a, v in slot.items()} for k, slot in by_cohort.items()}


def plot_roc(all_runs: dict[str, Path], out_dir: Path) -> Path | None:
    """Overlay ROC curves for every cohort across every run on one figure.

    Args:
        all_runs: Mapping of run tag -> run directory.
        out_dir: Directory to write the PNG into (created if missing).

    Returns:
        Path to the written PNG, or ``None`` if no run had raw predictions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    any_curve = False
    for run_tag, run_dir in all_runs.items():
        preds = _cohort_predictions(run_dir, run_tag)
        for (cohort), arrays in preds.items():
            roc = _cohort_roc(arrays["y_true"], arrays["y_score"])
            if roc is None:
                continue
            fpr, tpr, auc = roc
            ax.plot(fpr, tpr, label=f"{cohort} ({run_tag}) AUC={auc:.3f}")
            any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (per cohort, folds concatenated)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "roc_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_pr(all_runs: dict[str, Path], out_dir: Path) -> Path | None:
    """Overlay PR curves for every cohort across every run on one figure.

    Args:
        all_runs: Mapping of run tag -> run directory.
        out_dir: Directory to write the PNG into (created if missing).

    Returns:
        Path to the written PNG, or ``None`` if no run had raw predictions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    any_curve = False
    for run_tag, run_dir in all_runs.items():
        preds = _cohort_predictions(run_dir, run_tag)
        for (cohort), arrays in preds.items():
            pr = _cohort_pr(arrays["y_true"], arrays["y_score"])
            if pr is None:
                continue
            recall, precision, ap = pr
            ax.plot(recall, precision, label=f"{cohort} ({run_tag}) AP={ap:.3f}")
            any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (per cohort, folds concatenated)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "pr_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion(all_runs: dict[str, Path], out_dir: Path) -> list[Path]:
    """Render a confusion-matrix heatmap per run/cohort (all folds concatenated).

    Args:
        all_runs: Mapping of run tag -> run directory.
        out_dir: Directory to write the PNGs into (created if missing).

    Returns:
        Paths to the written PNGs (one per run that had raw predictions).
    """
    from sklearn.metrics import confusion_matrix

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for run_tag, run_dir in all_runs.items():
        preds = _cohort_predictions(run_dir, run_tag)
        if not preds:
            continue
        cohorts = sorted({c for _, c in preds})
        fig, axes = plt.subplots(1, len(cohorts), figsize=(4 * len(cohorts), 4), squeeze=False)
        for ax, cohort in zip(axes[0], cohorts, strict=False):
            arrays = preds[(run_tag, cohort)]
            cm = confusion_matrix(arrays["y_true"], arrays["y_pred"])
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(f"{cohort}")
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Confusion matrices ({run_tag}, folds concatenated)")
        fig.tight_layout()
        safe = run_tag.replace("/", "_").replace(":", "_")
        path = out_dir / f"confusion_{safe}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


@app.command()
def visualize(
    run_dirs: list[Path] = typer.Option(  # noqa: B008
        ...,
        "--dir",
        help="One or more experiment run directories (each with logs/*.jsonl).",
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Where PNGs are written (default: plots/ next to the first --dir).",
    ),
) -> None:
    """Render per-fold / grid-search / ROC / PR / confusion plots from run logs."""
    if not run_dirs:
        typer.echo("No --dir given.", err=True)
        raise typer.Exit(1)
    out_dir = output_dir or (run_dirs[0].parent / "plots")

    all_fold_rows: list[dict[str, Any]] = []
    all_grid_rows: list[dict[str, Any]] = []
    all_runs: dict[str, Path] = {}
    for run_dir in run_dirs:
        if not run_dir.exists():
            typer.echo(f"Run dir not found: {run_dir}", err=True)
            continue
        fold_rows, grid_rows, _run_started, label = collect_events(run_dir)
        for r in fold_rows:
            r["run"] = label
        for r in grid_rows:
            r["run"] = label
        all_fold_rows.extend(fold_rows)
        all_grid_rows.extend(grid_rows)
        all_runs[label] = run_dir
        typer.echo(
            f"loaded {run_dir}: {len(fold_rows)} fold events, {len(grid_rows)} grid summaries "
            f"(label={label})"
        )

    if not all_fold_rows:
        typer.echo("No fold_completed events found in any --dir.", err=True)
        raise typer.Exit(1)

    fold_df = pd.DataFrame(all_fold_rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scalar-metric plots (always in scope -- only need FoldCompletedEvent).
    p_bars = plot_per_fold_bars(fold_df, out_dir)
    typer.echo(f"wrote {p_bars}")
    p_grid = plot_grid_summary(fold_df, out_dir)
    typer.echo(f"wrote {p_grid}" if p_grid else "grid-search summary: skipped (no grid rows)")

    # Raw-prediction plots (need the .npz side-file; non-grid runs only).
    p_roc = plot_roc(all_runs, out_dir)
    typer.echo(f"wrote {p_roc}" if p_roc else "ROC curves: skipped (no npz predictions found)")
    p_pr = plot_pr(all_runs, out_dir)
    typer.echo(f"wrote {p_pr}" if p_pr else "PR curves: skipped (no npz predictions found)")
    p_cm = plot_confusion(all_runs, out_dir)
    for p in p_cm:
        typer.echo(f"wrote {p}")
    if not p_cm:
        typer.echo("confusion matrices: skipped (no npz predictions found)")

    typer.echo(f"done: {len(all_runs)} run(s) -> {out_dir}")


if __name__ == "__main__":
    app()
