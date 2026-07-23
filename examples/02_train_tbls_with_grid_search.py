"""Worked example 02: a small TBLS hyperparameter grid search on real data.

This is documentation-by-example: the shortest path from "I have the real
biomedical pkl" to "a ranked ``--grid`` sweep, exactly as the training CLI
produces". Like example 01 it is *not* a test -- it exists so a new user or
reviewer can see the intended grid-search usage pattern in under a minute.

It drives ``experiments/train.py``'s grid path *programmatically* (importing
and calling the same internals the test suite in
``tests/test_experiments_train.py`` calls) rather than shelling out to a
subprocess, so the call is debuggable and the ranked ``GridSummary`` rows are
available in-process to print.

Prerequisites:
    - Install the experiments-only dependencies (heavier than the published
      package)::

          uv sync --group experiments

    - The real dataset must already be present at
      ``experiments/datasets/biomedical_larger.pkl`` (git-ignored; see
      ``experiments/datasets/README.md``).

Expected runtime: a few seconds (1703 samples, 204 features; a tiny 2x2 grid
over ``n_map_trees``/``reg_param`` with 2-fold CV = 4 fits).

What it does, step by step:
    1. Load the ``"DM"`` cohort via :func:`experiments.train._load_cohorts`
       (the same loader the training CLI uses).
    2. Override :data:`experiments.train.TBLS_GRID` with a small 2x2 inline grid
       (``n_map_trees in {10, 20}``, ``reg_param in {1e-8, 1e-4}``) so the
       example finishes in seconds. The full grid lives in
       :mod:`experiments.hyperparams`; swap the override for
       ``hyperparams.TBLS_GRID`` to run the real 27-point sweep.
    3. Call :func:`experiments.train._run_grid` with a YAML-equivalent ``cfg``
       dict (model=TBLS with IFS weighting on, 2-fold CV, Lasso preprocessing),
       the same internal the ``--grid`` CLI flag calls.
    4. Print the ranked ``GridSummary`` rows (sorted by average balanced
       accuracy, descending) to stdout.

A :class:`experiments.evaluate.TBLSResultSaver` is reused (rather than
reimplementing output) so the per-point fold sheets and ranked summary are
written to a temporary directory exactly as the CLI would -- the ranked rows
printed below are the same rows it writes to the ``GridSummary`` sheet.

See ``docs/usage-tbls.md`` for the full TBLS API and
``docs/usage-experiments-cli.md`` for the training CLI (including ``--grid``).
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import time

# ``experiments/`` is not an installed package: its modules use script-style
# sibling imports (``from dataprocess import ...``), so both the repo root (for
# ``experiments.*`` namespace imports) and the ``experiments/`` directory itself
# (so those sibling imports resolve at import time) must be on ``sys.path``.
# This mirrors what ``tests/conftest.py`` does for the test suite.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.evaluate import TBLSResultSaver  # noqa: E402
import experiments.train as _train_mod  # noqa: E402
from experiments.train import _load_cohorts, _run_grid  # noqa: E402


def _format(value: object) -> str:
    """Format a metric value for the ranked table (None -> ``n/a``)."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    """Run a 2x2 TBLS grid search on the DM cohort and print ranked rows.

    Returns:
        0 on success, 1 if the dataset pkl is missing.
    """
    pkl_path = _REPO_ROOT / "experiments" / "datasets" / "biomedical_larger.pkl"
    if not pkl_path.exists():
        print(
            f"Dataset not found at {pkl_path}.\n"
            "Place the real .pkl under experiments/datasets/ (git-ignored); "
            "see experiments/datasets/README.md.",
            file=sys.stderr,
        )
        return 1

    # 1. Load the "DM" cohort as the training CLI does (single-view (X, y)).
    cohorts = _load_cohorts(pkl_path)
    if "DM" not in cohorts:
        print(f"Cohort 'DM' not found; available keys: {list(cohorts)}", file=sys.stderr)
        return 1
    cohort = cohorts["DM"]

    # 2. Override the model grid with a tiny 2x2 inline grid so the example
    #    finishes in seconds. ``_run_grid`` reads ``experiments.train.TBLS_GRID``
    #    from the module global at call time (see ``test_experiments_train.py``'s
    #    monkeypatch of the same attribute), so assigning it here is the
    #    example-script equivalent of the test's monkeypatch.
    _train_mod.TBLS_GRID = {"n_map_trees": [10, 20], "reg_param": [1e-8, 1e-4]}

    # 3. YAML-equivalent config: TBLS with IFS weighting on (the differentiator
    #    showcased in example 01), 2-fold CV, Lasso feature selection. This is
    #    the same shape of dict the CLI builds from default.yaml + overrides.
    cfg = {
        "model": {"name": "tbls", "use_if_weights": True},
        "cv": {"n_splits": 2, "random_state": 42},
        "preprocess": {"feature_selection": "lasso"},
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # The saver reuses the CLI's Excel writer; point it at a throwaway temp dir
    # (the ranked rows are printed below -- the Excel files are a side effect).
    out_dir = Path(tempfile.mkdtemp(prefix="tbls_grid_example_"))
    saver = TBLSResultSaver(
        dataset_name="biomedical_larger",
        timestamp=timestamp,
        key="DM",
        model_name="tbls",
        feature_selection="lasso",
        output_dir=str(out_dir),
    )

    # 4. Sweep the grid. ``_run_grid`` runs k-fold CV per grid point, writes the
    #    per-point fold sheets + a ranked ``GridSummary`` sheet via the saver,
    #    and returns the ranked rows (sorted by avg_balanced_accuracy desc).
    rows = _run_grid(
        cfg,
        cohort,
        dataset_name="biomedical_larger",
        key="DM",
        model_name="tbls",
        saver=saver,
    )

    # Print a compact ranked table of the same GridSummary rows.
    print("Worked example 02: TBLS grid search (cohort=DM, grid=2x2, 2-fold CV)")
    print(f"  Excel results written to: {out_dir}")
    print("  ranked by avg_balanced_accuracy (descending):")
    header = (
        f"  {'rank':>4} {'grid_idx':>8} {'n_map_trees':>11} {'reg_param':>10} "
        f"{'avg_acc':>8} {'avg_bal_acc':>11} {'avg_f1':>8} {'avg_auroc':>9}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for rank, row in enumerate(rows, start=1):
        print(
            f"  {rank:>4} {row['grid_idx']:>8} {row['n_map_trees']:>11} "
            f"{row['reg_param']:>10.0e} {_format(row.get('avg_accuracy')):>8} "
            f"{_format(row.get('avg_balanced_accuracy')):>11} "
            f"{_format(row.get('avg_f1_score')):>8} {_format(row.get('avg_auroc')):>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
