English | [简体中文](./internals.zh-CN.md)

# Pipeline internals (for maintainers)

This page is for someone who wants to **change or extend the pipeline itself**,
not just run it. For running, start at [index.md](index.md). This page assumes
you've already read the cli-*/outputs pages and now want the why-each-piece
view.

## Module map (`experiments/`)

| File | Role |
|---|---|
| `train.py` | Orchestrator: typer CLI, YAML loading, run/`cohort`/`fold` loop, model build, evaluation, Excel/log writing. The script-style sibling imports at the top (`from dataprocess import ...`) require `experiments/` on `sys.path` — `tests/conftest.py` sets this; CLI runs via `uv run experiments/train.py` work because `experiments/` is the script's own dir. |
| `classifiers.py` | The baseline factory `create_classifier(name, ...)`. Every supported baseline name has a small `if name == ...:` branch that builds an sklearn-wrapped estimator with class-balancing defaults. Soft dependencies (`xgboost`/`lightgbm`/`catboost`/`torch`) are imported lazily and raise `ImportError` only when the matching `name` is requested. |
| `dataprocess.py` | `DataLoader` — the pkl/CSV loader + feature selection + resampling. The legacy CSV pair path (`_load_csv`) uses `MultiLabelBinarizer`; the pkl path binarizes to `{0, 1}`. Tagging / CSV path is *intentionally* not what the binary TBLS pipeline uses — it's a legacy hook for multi-label work. |
| `evaluate.py` | `TBLSEvaluator` (scalar metrics, binary + multiclass dispatcher) + `TBLSResultSaver` (Excel writer). |
| `metrics_schema.py` | `MetricsDict` TypedDict — the canonical per-fold metrics schema. Shared by `evaluate.py` and `logging_schema.py` to avoid an import cycle. |
| `hyperparams.py` | Module-level Python-dict constants `TBLS_DEFAULTS`/`BLS_DEFAULTS`/`TBLS_GRID`/`BLS_GRID`/`CCA_*`/`GFCCA_*`. Not YAML (by design: values version-controlled + reviewed). |
| `logging_setup.py` | `configure_logging(output_dir, dataset, timestamp)`: dual loguru sinks (stdout INFO + JSONL DEBUG `serialize=True`) + the stdlib `InterceptHandler`. |
| `logging_schema.py` | The 5 event `TypedDict`s (`RunStartedEvent`, `FoldCompletedEvent`, `GridPointCompletedEvent`, `GridSummaryEvent`, `RunFinishedEvent`). |
| `multiview.py` | `MultiViewDataLoader` + `load_multiview_cohort` + `fuse_views` (single-view `_cross_validate` path bypasses this entirely). |
| `run_resolution.py` | `resolve_run_dir(run_arg) / cohort_excel_dir(run_dir, cohort)` — the canonical `--dir` resolution rule (run-name layer vs run-name/timestamp vs ill-formed). Shared by `visualize.py` and `compare.py`. |
| `visualize.py` | `/visualize` CLI: parses JSONL events + npz side-files, renders matplotlib PNGs. |
| `compare.py` | `/compare` CLI: parses JSONL fold events across runs, writes `comparison.xlsx` with `mean (std)` cells, bolds the per-(cohort, metric) best by `METRIC_DIRECTION`. |
| `smoke_run.py` | The single-k-split sanity check + a `_extract_xy` helper reused by the example scripts and the real-dataset tests. |

## Where `model.name` is dispatched

`train.py::_build_model(model_cfg, grid_point=None)`:

```python
if name == "tbls":  defaults = TBLS_DEFAULTS    ; cls = TBLS
elif name == "bls": defaults = BLS_DEFAULTS     ; cls = BroadLearningSystem
else:  return create_classifier(name, random_state=..., **YAML_kwargs)
```

For the `tbls`/`bls` tier, YAML keys are filtered against the constructor
signature (legacy `map_num`/`enhance_num` → `n_map_trees`/`n_enhance_trees`),
then `grid_point` overrides win. For baselines, YAML keys go directly to
`create_classifier` as `**kwargs` (no signature filtering — if you pass an
invalid kwarg for that sklearn estimator, the constructor will raise), and
`grid_point` is **ignored** (baselines don't participate in `--grid`).

## Two-tier grid resolution (`_resolve_grid`)

- **Default**: `TBLS_GRID` for `tbls`, `BLS_GRID` for `bls`, *nothing* for a
  baseline.
- **YAML `grid:`**: if `grid:` is in the cfg, the resolved grid is `default`
  copied, then every axis named in YAML *replaces* the same-named axis's list.
  Axes only in YAML but not in the default are added (so you can sweep a
  baseline by giving it only a YAML `grid:`).
- Baselines with no YAML `grid:` raise `ValueError("No default grid for ...")`
  *inside* `_resolve_grid` — but the caller in `train.py` checks first and
  falls back to a single-k-fold-with-warning rather than ever calling
  `_resolve_grid` for a bare `--grid` baseline, so the user path is graceful.

## `--dir` resolution (`run_resolution.resolve_run_dir`)

The rule is mechanically enforced:

1. If `run_arg.name` matches `^\d{8}_\d{6}$` → treat `run_arg` as the timestamp
   layer (direct use). Verify a `logs/` subdir exists; verify no
   timestamp-under-timestamp nesting (would mean too-deep).
2. Otherwise → search `run_arg/*` for `YYYYMMDD_HHMMSS` subdirs, pick the
   lex-max (newest), verify its `logs/` subdir exists.
3. Any other shape (shallower than run-name, deeper than `<ts>/logs`, or a
   timestamp dir whose parent isn't a run-name layer) raises a typed
   exception with a one-line diagnostic.

`cohort_excel_dir(run_dir, cohort)` returns `run_dir.parent / cohort /
run_dir.name` (sibling cohort dir at the same timestamp) and raises if that
path doesn't exist — so `compare.py` refuses to silently pull a mismatching
timestamp cohort dir.

## Raw-prediction persistence (`_cross_validate`)

Only for non-grid folds, the per-fold `y_true`/`y_pred`/`y_score` are
accumulated into `preds[{cohort}_fold{N}_{y_true,y_pred,y_score}]` then
`np.savez`'d at the end of `_cross_validate`. Grid runs pass
`predictions_npz=None`, so the npz is skipped — see Plan 02/06 records in git
log (~c811812 region) for the size reason. The `fold_completed` event's
`predictions_file` field carries the npz *name* (no path) so consumers load it
relative to the JSONL's own `logs/` dir.

## What's NOT happening (so you don't go looking)

- No probability calibration. `TBLS.predict_proba` is the closed-form ridge
  output softmax-transformed; the `0.5` score-density plateau producing the
  PR cliff is a model artifact, not a pipeline artifact. See
  [figures-and-calibration.md](figures-and-calibration.md).
- `--grid` does not sweep CCA/GFCCA fusion axes. (Documented scope limit.)
- The legacy CSV route in `DataLoader` does not interact with the binary
  pipeline; do not try to make it interact.
- `run_resolution` deliberately doesn't accept a bare directory of multiple
  runs (`examples/runs`) — every run must be addressed individually. Batch is
  `--config-dir` at `train.py`-level, not at run-resolution level.