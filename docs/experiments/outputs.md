# What lands on disk after a run

This is the reference for every file written by `train.py`, the Excel sheet
layout, the JSONL event schema, and the `.npz` side-file schema.

## Directory layout

`train.py` writes (relative to `output_dir`, defaulted to `results_dir/`):

```
{output_dir}/{run_name}/{timestamp}/                       <- run dir
    logs/{dataset}_{timestamp}.jsonl                       <- structured log
    logs/{dataset}_{timestamp}_{cohort}_predictions.npz    <- raw predictions
                                                             (non-grid runs only)

{output_dir}/{run_name}/{cohort}/{timestamp}/              <- cohort dir
    {cohort}_{model_name}_FS-{fs}_RS-{rs}.xlsx             <- per-cohort Excel
```

`{timestamp}` is `time.strftime("%Y%m%d_%H%M%S")`, shared by both branches
(the run dir and the cohort dir are timestamp siblings under
`{run_name}`). The same `{run_name}` appears as the stem and as the YAML
label; `examples/runs/`/`plots/`/`examples/comparison/` are git-ignored.

## Excel (`{cohort}_{model_name}_FS-..._RS-..._xlsx`)

One file per cohort. `TBLSResultSaver` accumulates sheets as a run progresses.

### Non-grid runs

| Sheet | Rows |
|---|---|
| `{model}_Details` | one row per CV fold, with every per-fold metric |
| `{model}_Summary` | one row, cross-fold averages (keys prefixed `avg_`) + `{key: cohort}` |
| `{sheet}_Meta` | one row, `Feature_Selection`/`Resampling_Method` metadata |

### `--grid` runs

| Sheet | Rows |
|---|---|
| `Grid_{i:03d}` per grid point | a `_cross_validate` fold-level dump for grid point `i` |
| `GridSummary` | one row per grid point, ranked by `avg_balanced_accuracy` descending; swept hyperparams + every `avg_*` metric + `rank` + `is_winner` (**row 1 = winner**) |
| `GridSearchLog` | same as `GridSummary` but flat (chronological pre-rank order) — read as the search log |
| `{sheet}_Meta` | feature selection + resampling metadata |

## JSONL log (`logs/{dataset}_{timestamp}.jsonl`)

Two sinks are configured at run start by `experiments/logging_setup.py`:
human-readable stdout (INFO, colorized) and this JSONL file (DEBUG,
`serialize=True`). Each line is one record: loguru's metadata under
`record.{text, level, time, function, line, ...}` and any bound event payload
under `record.extra`.

Events (`event` is the discriminator field in `record.extra`):

### `run_started` (once per run)
| Field | Type | Notes |
|---|---|---|
| `dataset` | str | dataset stem |
| `model` | str | model name (`"tbls"`, `"bls"`, or a baseline name) |
| `fusion_method` | str \| None | only for multi-view cohorts; `None` otherwise |
| `grid` | bool | was `--grid` passed |
| `run_name` | str \| None | only when YAML `run_name:` is set; absent otherwise |

### `fold_completed` (per fold, per cohort)
| Field | Type | Notes |
|---|---|---|
| `dataset`, `cohort_key` | str | — |
| `fold`, `n_splits` | int | 1-indexed fold; total fold count |
| `metrics` | `MetricsDict` | every per-fold scalar metric (see "Metric keys" below) |
| `grid_idx` | int \| None | 1-indexed grid-point number (only under `--grid`; else `None`) |
| `grid_params` | dict \| None | the swept hyperparams (only under `--grid`; else `None`) |
| `predictions_file` | str \| None | the `.npz` side-file name (only for non-grid folds; else `None`) |

### `grid_point_completed` (per swept grid point, under `--grid`)
| Field | Type | Notes |
|---|---|---|
| `dataset`, `cohort_key` | str | — |
| `grid_idx`, `n_grid_points` | int | 1-indexed; total points |
| `grid_params` | dict | the hyperparams of this point |
| `metrics` | dict | averaged-over-folds metrics (prefix `avg_`) |

### `grid_summary` (once per cohort, after `--grid` ranks)
| Field | Type | Notes |
|---|---|---|
| `dataset`, `cohort_key` | str | — |
| `winner_params` | dict | winner's hyperparam row |
| `winner_metric` | float | winner's `avg_balanced_accuracy` |
| `n_grid_points` | int | total swept points |

### `run_finished` (once per run)
| Field | Type | Notes |
|---|---|---|
| `dataset` | str | — |
| `duration_seconds` | float | wall-clock run duration |

Stdlib `logging` is intercepted too (an `InterceptHandler` forwards
`logging.getLogger(...).warning(...)` calls into the same JSONL), so for
example `experiments.evaluate`'s probability-metric warnings appear here too.

### Metric keys in `fold_completed.metrics`

The schema is the `MetricsDict` TypedDict in
`experiments/metrics_schema.py`. Binary path returns:
`accuracy`, `precision`, `recall`, `f1_score`, `hamming_loss`, `specificity`,
`negative_predictive_value`, `balanced_accuracy`, `gmean`, `mcc`,
`cohen_kappa`, and (when `y_score` is present) `auroc`, `auprc`,
`optimal_threshold`, `log_loss`, `brier_score`. Multiclass path returns the
same except binary-only keys (`auprc`/`optimal_threshold`/`log_loss`/
`brier_score`) are absent; it also gains `precision_weighted`/`recall_weighted`/
`f1_weighted` and uses `balanced_accuracy_score` directly.

Probability-based keys (`auroc`/...) degrade to `None` when `y_score` is
missing or the underlying sklearn call raises on a degenerate fold.

## `.npz` predictions side-file

For non-grid folds, a per-cohort `.npz` file is written under
`logs/` named `{dataset}_{timestamp}_{cohort}_predictions.npz`. Each file
holds, per fold:

| Key | dtype/shape |
|---|---|
| `{cohort}_fold{N}_y_true` | `int64` shape `(n_test,)` |
| `{cohort}_fold{N}_y_pred` | `int64` shape `(n_test,)` — `model.predict(X_te)` |
| `{cohort}_fold{N}_y_score` | `float32` shape `(n_test, n_classes)` — `model.predict_proba(X_te)` |

`y_score` is a 2-D probability matrix, NOT a 1-D positive-class vector.
Readers wanting the binary positive class must slice `y_score[:, 1]` —
`visualize.py` and `compare.py` already do this.

Grid runs do **not** write a side-file (a 27 × n_folds × n_cohorts sweep would
blow the file up). `fold_completed.predictions_file` is `None` on grid runs
for that reason.