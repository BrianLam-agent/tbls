# Worked examples

Runnable, heavily-commented scripts that show the *currently-accepted*
capabilities of the `tbls` package against the real `biomedical_larger.pkl`
dataset, end to end. These are documentation-by-example: the shortest path
from "I have the real pkl" to "TBLS fitted / a ranked grid sweep," so a new
user or reviewer can see the intended usage pattern in under a minute. They
are **not** tests — the test suite in `tests/` already covers correctness.

The `examples/` directory is *not* part of the published wheel — like
`experiments/`, it only ships in the source repo.

## Prerequisites

1. Install the experiments-only dependency group (heavier than the published
   package; brings `pandas`, `imbalanced-learn`, `openpyxl`, `typer`, ...):

   ```bash
   uv sync --group experiments
   ```

2. The real dataset must already be present at
   `experiments/datasets/biomedical_larger.pkl`. It is **git-ignored** (see
   `experiments/datasets/README.md`); the examples fail with a clear message
   if it is missing.

## Example 01 — a single TBLS training run

```bash
uv run --group experiments python examples/01_train_tbls.py
```

Loads the `"DM"` cohort, a single stratified train/test split (no CV),
standardizes + Lasso-selects features fit on the train split only, fits one
`TBLS` with Intuitionistic Fuzzy Set sample weighting on (`use_if_weights=True`,
the TBLS differentiator versus a plain Broad Learning System), and prints
held-out accuracy / balanced accuracy / macro F1.

Expected output (approximate — varies slightly across machines and BLAS):

```text
Worked example 01: single TBLS run (cohort=DM)
  train=1362 test=341 features_in=204 features_selected=62
  accuracy          = 0.9179
  balanced_accuracy = 0.9099
  macro_f1          = 0.8943
```

## Example 02 — a TBLS hyperparameter grid search

```bash
uv run --group experiments python examples/02_train_tbls_with_grid_search.py
```

Drives `experiments/train.py`'s `--grid` path **programmatically** (importing
and calling the same internal `_run_grid` the test suite calls, rather than
shelling out to a subprocess). Runs a small 2x2 grid (`n_map_trees in {10, 20}`,
`reg_param in {1e-8, 1e-4}`) with 2-fold CV on the `"DM"` cohort and prints
the ranked `GridSummary` rows (sorted by average balanced accuracy,
descending). A `TBLSResultSaver` writes the per-point fold sheets and ranked
summary to a throwaway temp directory; the printed rows are exactly the
`GridSummary` sheet contents.

Expected output (approximate — varies slightly across machines and BLAS):

```text
Worked example 02: TBLS grid search (cohort=DM, grid=2x2, 2-fold CV)
  Excel results written to: <temp dir>
  ranked by avg_balanced_accuracy (descending):
  rank grid_idx n_map_trees  reg_param  avg_acc avg_bal_acc   avg_f1 avg_auroc
  ----------------------------------------------------------------------------
     1        1          10      1e-08   0.9184      0.9026   0.8412    0.8790
     2        2          10      1e-04   0.9172      0.9020   0.8394    0.8800
     3        4          20      1e-04   0.9049      0.8836   0.8142    0.8412
     4        3          20      1e-08   0.9037      0.8795   0.8105    0.8343
```

## Full API / CLI reference

These examples deliberately do not repeat the documented API/CLI surface:

- [`docs/usage-tbls.md`](../docs/usage-tbls.md) — the full `TBLS` tutorial
  (parameters, IFS / graph-Laplacian regularization, incremental layers,
  reproducibility, performance notes).
- [`docs/usage-experiments-cli.md`](../docs/usage-experiments-cli.md) — running
  the `experiments/` training CLI (incl. `--grid`) and smoke-check script
  against real datasets.

For the minimal runnable precedent these examples mirror, see
[`experiments/smoke_run.py`](../experiments/smoke_run.py).