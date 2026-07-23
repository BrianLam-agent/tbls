"""Typed event schemas for the structured (JSONL) experiment log.

These ``TypedDict``s define the shape of every structured record emitted by
``experiments/train.py`` and persisted by
:func:`experiments.logging_setup.configure_logging` to the per-run JSONL file
under ``{output_dir}/logs/{dataset}_{timestamp}.jsonl``. Each event is emitted
at its call site via ``logger.bind(**event).info(event["event"])``; loguru's
``serialize=True`` file sink writes the whole record (including the bound
``event`` dict under ``record["extra"]``) as one JSON object per line, which
``experiments/visualize.py`` reads back out.

The ``FoldCompletedEvent`` carries only scalar metrics (to keep the JSONL
small). Raw per-fold predictions needed for ROC/PR/confusion-matrix plots are
written to a side-file ``logs/{dataset}_{timestamp}_predictions.npz`` and
referenced by ``predictions_file``; this side-file is produced for non-grid
runs only (grid sweeps would explode its size) -- see
``docs/usage-experiments-cli.md``.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from metrics_schema import MetricsDict


class RunStartedEvent(TypedDict, total=False):
    """Emitted once at the start of a ``train.py`` run.

    Attributes:
        event: Literal event discriminator (``"run_started"``).
        dataset: Dataset stem loaded by the run.
        model: Model name (``"tbls"``, ``"bls"``, or any baseline name from
            :func:`experiments.classifiers.create_classifier`).
        fusion_method: Active fusion method for multi-view cohorts, or
            ``None`` for single-view runs.
        grid: Whether ``--grid`` is sweeping the hyperparameter grid.
        run_name: Optional human-chosen experiment name set in YAML as
            ``run_name:`` (preferred for figure labels over the auto-generated
            ``{model}_{dataset}/{timestamp}`` fallback). Absent when not set.
    """

    event: Literal["run_started"]
    dataset: str
    model: str
    fusion_method: str | None
    grid: bool
    run_name: str | None


class FoldCompletedEvent(TypedDict):
    """Emitted per fold (per cohort) inside ``_cross_validate``.

    Attributes:
        event: Literal event discriminator (``"fold_completed"``).
        dataset: Dataset stem.
        cohort_key: Sub-dataset key (or ``"single"``).
        fold: 1-indexed fold number.
        n_splits: Total number of folds.
        metrics: Scalar per-fold metrics (no probability curves).
        grid_idx: 1-indexed grid point when running under ``--grid``, else
            ``None``.
        grid_params: The grid point's hyperparameters when under ``--grid``,
            else ``None``.
        predictions_file: Filename of the ``.npz`` side-file holding this
            run's raw per-fold ``y_true``/``y_pred``/``y_score`` arrays, for
            non-grid runs only; ``None`` for grid runs.
    """

    event: Literal["fold_completed"]
    dataset: str
    cohort_key: str
    fold: int
    n_splits: int
    metrics: MetricsDict
    grid_idx: int | None
    grid_params: dict[str, object] | None
    predictions_file: str | None


class GridSummaryEvent(TypedDict):
    """Emitted once per cohort after ``_run_grid`` ranks its grid points.

    Attributes:
        event: Literal event discriminator (``"grid_summary"``).
        dataset: Dataset stem.
        cohort_key: Sub-dataset key.
        winner_params: Hyperparameters of the top-ranked grid point.
        winner_metric: The top-ranked grid point's ``avg_balanced_accuracy``.
        n_grid_points: Total number of swept grid points.
    """

    event: Literal["grid_summary"]
    dataset: str
    cohort_key: str
    winner_params: dict[str, object]
    winner_metric: float
    n_grid_points: int


class RunFinishedEvent(TypedDict):
    """Emitted once at the end of a ``train.py`` run.

    Attributes:
        event: Literal event discriminator (``"run_finished"``).
        dataset: Dataset stem.
        duration_seconds: Wall-clock run duration in seconds.
    """

    event: Literal["run_finished"]
    dataset: str
    duration_seconds: float
