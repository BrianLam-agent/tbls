English | [简体中文](./cli-compare.zh-CN.md)

# `experiments/compare.py` — cross-run comparison Excel

`compare.py` reads one or more run directories (the same `--dir` rule as
`visualize.py`) and writes a single `comparison.xlsx` summarizing every run's
per-cohort metrics in paper-table form. Run with `uv run --group experiments`.

## Options

### `--dir DIR` (repeatable; required at least once)
- **What**: a run directory, in either form (auto-detected):
  - run-name layer, e.g. `examples/runs/TBLS Full` —
    auto-picks the newest `YYYYMMDD_HHMMSS` timestamp subdirectory;
  - run-name/timestamp layer, e.g.
    `examples/runs/TBLS Full/20260724_074140` — used directly.
- **Conflict**: deeper / shallower / non-timestamp paths error out exactly as
  in [cli-visualize.md](cli-visualize.md).
- **Duplicate labels**: two `--dir` arguments that resolve to the same run
  name raise `ValueError: Duplicate run label ...`.

### `--output-dir DIR`
- **What**: where `comparison.xlsx` is written.
- **Default**: `examples/comparison`.

### `--no-std`
- **What**: drop the `(std)` term from each cell; write bare means instead of
  `mean (std)`.
- **Default**: off (every cell is `mean (std)`).

## Output: `comparison.xlsx`

Contains one Excel sheet per cohort plus a `README` sheet:

| Sheet | Contents |
|---|---|
| `README` | layout + per-metric direction table (which side is "best") |
| `<cohort>` | rows = runs (sorted), columns = 15 scalar metrics |

Each cell is `mean (std)` across CV folds for that (cohort, run, metric), e.g.
`0.9237 (0.0112)`. With `--no-std` cells become bare means like `0.9237`.

### The 15 metrics + their direction (drives the bold)

`compare.py` bolds the **best** run per (cohort, metric). "Best" depends on
whether higher or lower is better for that metric:

| Metric | Direction (bold the run with the ...) |
|---|---|
| `balanced_accuracy` | highest mean |
| `accuracy` | highest |
| `f1_score` | highest |
| `mcc` | highest |
| `cohen_kappa` | highest |
| `auroc` | highest |
| `auprc` | highest |
| `recall` | highest |
| `specificity` | highest |
| `precision` | highest |
| `negative_predictive_value` | highest |
| `gmean` | highest |
| `hamming_loss` | lowest (lower is better) |
| `log_loss` | lowest |
| `brier_score` | lowest |

A run that did not produce a cohort leaves that cell blank (not 0, not NaN) so
missing cohorts are visually obvious.

## Data source

`compare.py` parses the `fold_completed` events from each run's
`logs/{dataset}_{timestamp}.jsonl` (not the Excel files) and computes
mean/std across the folds for each scalar metric it finds there. The metric
schema is documented in [outputs.md](outputs.md).

## Typical invocations

```bash
# 5-run ablation: write mean ± std comparison, best bolded per metric
uv run --group experiments python experiments/compare.py \
    --dir "examples/runs/TBLS" \
    --dir "examples/runs/TBLS IFS" \
    --dir "examples/runs/TBLS Graph" \
    --dir "examples/runs/TBLS Full" \
    --dir "examples/runs/Logistic Regression" \
    --output-dir examples/comparison

# Same but numeric-only (no std, easier in a spreadsheet): add --no-std
uv run --group experiments python experiments/compare.py \
    --dir "examples/runs/TBLS" \
    --dir "examples/runs/Logistic Regression" --no-std
```

## Common errors

- **`--dir ... has no YYYYMMDD_HHMMSS timestamp subdirectory`** — same as
  visualize.py: you gave a run-name layer with no timestamp subdir under it.
- **`Per-cohort Excel dir not found: ...`** — `compare.py` looks for the
  sibling cohort Excel directory under the same timestamp the JSONL run
  directory used; if you moved files around after a run, that sibling may be
  gone. Re-run `train.py` to rebuild, or point at a different timestamp.