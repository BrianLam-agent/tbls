"""Cross-run comparison Excel CLI.

Reads one or more experiment run directories (each resolved by
:mod:`experiments.run_resolution` -- give the run-name layer to auto-pick the
newest timestamp, or the run-name/<timestamp> layer explicitly) and writes a
single ``comparison.xlsx`` summarising every run's per-cohort metrics across
all the provided runs, with the best value per (cohort, metric) bolded.

Each cell holds the metric's mean across folds followed by its standard
deviation in parentheses, e.g. ``0.9237 (0.0112)`` -- the standard
"mean ± std" presentation for paper tables. ``--no-std`` drops the std term
if you want bare means.

Metric direction (which value is "best") is internal to this module so the
bold picks the right one for each metric (``auroc``/``balanced_accuracy`` →
higher is better; ``log_loss``/``brier_score``/``hamming_loss`` → lower is
better).

Run with::

    uv run --group experiments python experiments/compare.py \
        --dir examples/runs/TBLS --dir "examples/runs/TBLS Full" \
        --dir "examples/runs/Logistic Regression" \
        --output-dir examples/comparison

produces ``examples/comparison/comparison.xlsx`` (one sheet per cohort, plus a
``README`` sheet describing the layout).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np
from openpyxl.styles import Font
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_resolution import resolve_run_dir  # noqa: E402

# Metrics lifted from FoldCompletedEvent.metrics (scalar ones only -- those that
# are single floats per fold, so mean/std are meaningful). Direction controls
# which side of the mean range is "best" and therefore gets bolded.
# Which side of the mean range is "best" for each metric (drives the bold).
METRIC_DIRECTION: dict[str, str] = {
    "accuracy": "higher",
    "balanced_accuracy": "higher",
    "precision": "higher",
    "recall": "higher",
    "f1_score": "higher",
    "specificity": "higher",
    "negative_predictive_value": "higher",
    "gmean": "higher",
    "mcc": "higher",
    "cohen_kappa": "higher",
    "auroc": "higher",
    "auprc": "higher",
    "hamming_loss": "lower",
    "log_loss": "lower",
    "brier_score": "lower",
}

ORDERED_METRICS: list[str] = [
    "balanced_accuracy",
    "accuracy",
    "f1_score",
    "mcc",
    "cohen_kappa",
    "auroc",
    "auprc",
    "recall",
    "specificity",
    "precision",
    "negative_predictive_value",
    "gmean",
    "hamming_loss",
    "log_loss",
    "brier_score",
]


def _collect_fold_metrics(run_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Parse every ``logs/*.jsonl`` under ``run_dir`` into ``{cohort: {metric: [folds]}}``.

    Args:
        run_dir: Resolved timestamped run directory (holds ``logs/``).

    Returns:
        ``{cohort_key: {metric_name: [per-fold float values]}}`` for every
        scalar metric present in ``FoldCompletedEvent.metrics``.
    """
    by_cohort: dict[str, dict[str, list[float]]] = {}
    for jl in sorted(run_dir.glob("logs/*.jsonl")):
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
                cohort = extra.get("cohort_key", "?")
                metrics = extra.get("metrics", {}) or {}
                slot = by_cohort.setdefault(cohort, {})
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        slot.setdefault(name, []).append(float(value))
    return by_cohort


def _format_cell(mean: float, std: float | None, with_std: bool) -> str:
    """Format one comparison cell as ``mean`` or ``mean (std)``.

    Args:
        mean: Mean across folds.
        std: Standard deviation across folds, or ``None`` to skip.
        with_std: If True, append ``(std)``; if False, return just ``mean``.

    Returns:
        The formatted cell string.
    """
    if not with_std or std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ({std:.4f})"


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return ``(mean, std)`` of a list of fold values.

    Args:
        values: Per-fold float values.

    Returns:
        ``(mean, std)``; ``std`` is the population std (ddof=0) for N folds.
    """
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _collect_all_runs(
    run_args: list[Path],
) -> tuple[dict[str, dict[str, dict[str, list[float]]]], dict[str, str]]:
    """Resolve and parse every ``--dir`` into ``(by_run, run_to_label)``.

    Args:
        run_args: Raw ``--dir`` arguments (run-name layer or run-name/<ts>).

    Returns:
        ``({run_label: {cohort: {metric: [folds]}}}, {run_label: resolved_path})``.
        The run label is the run-name directory's name (e.g. ``"TBLS Full"``),
        which is what the YAML ``run_name:`` produced.

    Raises:
        Propagates :func:`resolve_run_dir` errors for invalid ``--dir`` depth.
    """
    by_run: dict[str, dict[str, dict[str, list[float]]]] = {}
    run_to_path: dict[str, str] = {}
    for arg in run_args:
        run_dir = resolve_run_dir(arg)
        label = run_dir.parent.name  # the run_name stem (e.g. "TBLS Full")
        if label in by_run:
            raise ValueError(f"Duplicate run label {label!r} (from {arg}); drop a duplicate --dir.")
        by_run[label] = _collect_fold_metrics(run_dir)
        run_to_path[label] = str(run_dir)
    return by_run, run_to_path


def _metrics_to_show(by_run: dict[str, dict[str, dict[str, list[float]]]]) -> list[str]:
    """Return the metrics present across runs, in their canonical order."""
    present: set[str] = set()
    for cohorts in by_run.values():
        for fold_metrics in cohorts.values():
            present.update(fold_metrics.keys())
    return [m for m in ORDERED_METRICS if m in present]


def _bold_winners_in_sheet(ws, metrics: list[str], n_runs: int) -> None:
    """Bold the best run cell per metric column in an openpyxl worksheet.

    Each sheet has the layout:
        col 1: "run"
        col 2..: one per metric (header row), then one row per run with the
                 ``mean (std)`` STRING.
    The best mean per metric (highest for "higher"-direction metrics, lowest
    for "lower") is found by parsing the leading ``mean`` from the formatted
    string, then that cell gets bold. ``avg_``-prefixed columns aren't present
    here (this is the comparison layout, not the GridSummary layout).

    Args:
        ws: An openpyxl worksheet with the comparison layout.
        metrics: Metric basename per column (matches ``METRIC_DIRECTION`` keys).
        n_runs: Number of run rows (rows 2..n_runs+1).
    """
    for col_idx, metric in enumerate(metrics, start=2):
        direction = METRIC_DIRECTION.get(metric)
        if direction is None:
            continue
        best_mean: float | None = None
        best_row: int | None = None
        for row in range(2, n_runs + 2):
            cell = ws.cell(row=row, column=col_idx)
            text = cell.value
            if not isinstance(text, str):
                continue
            try:
                mean = float(text.split()[0])
            except (ValueError, IndexError):
                continue
            if (
                best_mean is None
                or (direction == "higher" and mean > best_mean)
                or (direction == "lower" and mean < best_mean)
            ):
                best_mean, best_row = mean, row
        if best_row is not None:
            ws.cell(row=best_row, column=col_idx).font = Font(bold=True)


@app.command()
def compare(
    run_dirs: list[Path] = typer.Option(  # noqa: B008
        ...,
        "--dir",
        help="One or more run directories -- the run-name layer (auto-picks "
        "newest timestamp) OR run-name/<timestamp> (used directly). Anything "
        "deeper, shallower, or non-timestamp errors out.",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("examples/comparison"),
        "--output-dir",
        help="Where comparison.xlsx is written (default: examples/comparison).",
    ),
    no_std: bool = typer.Option(
        False,
        "--no-std",
        help="Drop the (std) term; write bare means instead of mean (std).",
    ),
) -> None:
    """Write comparison.xlsx across multiple runs, best value bolded per metric."""
    if not run_dirs:
        typer.echo("No --dir given.", err=True)
        raise typer.Exit(1)

    by_run, run_to_path = _collect_all_runs(run_dirs)
    metrics = _metrics_to_show(by_run)

    # Cohorts: union across runs, sorted. A run missing a cohort leaves blank
    # cells in that cohort's sheet (rather than silently dropping the run).
    cohorts: set[str] = set()
    for fold_metrics_by_cohort in by_run.values():
        cohorts.update(fold_metrics_by_cohort.keys())
    sorted_cohorts = sorted(cohorts)
    sorted_runs = sorted(by_run.keys())

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comparison.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        # README sheet: layout + metric directions, so the reader knows the
        # bold = "best for that metric" without re-reading the source.
        readme_rows = [
            {
                "what": "Each table cell",
                "means": "mean across CV folds (std) for that run+cohort+metric",
            },
            {
                "what": "Bold cell",
                "means": "best run for that metric on that cohort (higher or lower per METRIC_DIRECTION)",
            },
            {
                "what": "Sheet per cohort",
                "means": "one sheet per cohort key (rows = runs, columns = metrics)",
            },
            {"what": "Missing run+cohort", "means": "blank (the run did not produce that cohort)"},
        ]
        for m in metrics:
            readme_rows.append(
                {"what": m, "means": f"direction={METRIC_DIRECTION.get(m, 'higher')}"}
            )
        pd.DataFrame(readme_rows).to_excel(writer, sheet_name="README", index=False)

        for cohort in sorted_cohorts:
            rows: list[dict[str, str]] = []
            for run_label in sorted_runs:
                per_run = by_run[run_label].get(cohort, {})
                row: dict[str, str] = {"run": run_label}
                for metric in metrics:
                    fold_values = per_run.get(metric)
                    if fold_values:
                        mean, std = _mean_std(fold_values)
                        row[metric] = _format_cell(mean, std, with_std=not no_std)
                    else:
                        row[metric] = ""
                rows.append(row)
            df = pd.DataFrame(rows, columns=["run", *metrics])
            # Sheet names: openpyxl forbids ``/`` ``?`` ``*`` ``[`` ``]`` ``:`` ``\``.
            safe_sheet = re.sub(r"[\[\]\*?:\/]", "_", cohort)
            df.to_excel(writer, sheet_name=safe_sheet, index=False)
        # Bold the winner per metric column per cohort sheet (post-write).
        for cohort in sorted_cohorts:
            safe_sheet = re.sub(r"[\[\]\*?:\/]", "_", cohort)
            ws = writer.sheets[safe_sheet]
            _bold_winners_in_sheet(ws, metrics, n_runs=len(sorted_runs))

    typer.echo(
        f"wrote {out_path} ({len(sorted_runs)} runs, {len(sorted_cohorts)} cohorts, "
        f"{len(metrics)} metrics; bold = best per (cohort, metric))"
    )
    for run_label, run_path in run_to_path.items():
        typer.echo(f"  {run_label}: {run_path}")


if __name__ == "__main__":
    app()
