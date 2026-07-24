English | [简体中文](./usage-experiments-cli.zh-CN.md)

# Running the experiments CLI (training on real datasets)

`experiments/` is the training/evaluation/comparison pipeline used to run
`tbls` estimators against real datasets on your own machine. It is **not** part
of the published `tbls` package (it depends on `pandas`, `imbalanced-learn`,
`xgboost`, `openpyxl`, `loguru`, `matplotlib`, `typer`, `pyyaml` — heavy,
opinionated dependencies that would weigh down every `pip install tbls`
user). See [`architecture.md`](./architecture.md#3-why-the-packageexperiments-split)
for the rationale. For a runnable ground-up tutorial see
[`../examples/README.md`](../examples/README.md); this document is the full
reference for every option, every output file, and every piece of behavior.

## Setup

```bash
git clone https://github.com/BrianLam-agent/tbls.git
cd tbls
uv sync --group dev --group experiments
```

The exact `experiments` dependency group is declared in `pyproject.toml`
(`[dependency-groups] experiments = [...]`). With `uv`, `--group experiments`
installs it; the dev group (`pytest`/`ruff`/`mypy`) is also recommended
locally.

## Datasets

Place dataset files under `experiments/datasets/` (this directory is
git-ignored — see
[`experiments/datasets/README.md`](../experiments/datasets/README.md)). The
training CLI loads them via `experiments/dataprocess.py::DataLoader`.

### Supported loaders (auto-chosen by file presence)

`DataLoader` tries `.csv` first, falling back to `.pkl`:

1. **CSV + label CSV pair** — if `{dataset}_data.csv` and `{dataset}_label.csv`
   both exist, it loads those via `np.loadtxt` (float32 X, int32 labels). This
   path was kept for the legacy multi-label workflow; it does **not** drop
   label `-1` or binarize — it uses `MultiLabelBinarizer` instead. Do **not**
   use this path for the `TBLS` binary-classification experiments; it is a
   legacy data-ingestion hook.
2. **Pickle (default for the binary `tbls` pipeline)** — if
   `{dataset}.pkl` exists, `joblib.load` reads it as one of:

   - a flat `{"data": X, "target": y}` dict (reported under key `"single"`),
   - a multi-key dict of such sub-datasets, each processed independently
     keyed by its dict key (e.g. one file holding several disease cohorts
     `{"DM": {...}, "CKD": {...}, ...}`),
   - a multi-view dict (keyed by its dict key) with `{"views": {...},
     "target": y}` instead of `"data"` — auto-detected for multi-view fusion
     (see [`usage-multiview-fusion.md`](./usage-multiview-fusion.md)).

   The pkl path preprocesses the labels canonically: samples with `y == -1`
   are dropped, labels binarize to `{0, 1}` via `(y > 0).astype(int)`,
   `dtype=object` feature matrices are coerced to `float64`, and
   `NaN`/`Inf` are zeroed. This is the same normalization
   `experiments/smoke_run.py::_extract_xy` uses.

## Minimal sanity check: `smoke_run.py`

`experiments/smoke_run.py` is the fastest way to confirm a dataset loads and
`TBLS` fits + predicts sanely on it (small model, one train/test split,
assertions: `predict_proba` finite, row-wise sum ≈ 1, non-single-class
predictions):

```bash
uv run --group experiments python experiments/smoke_run.py
```

```
TBLS smoke check OK | key=DM acc=0.9208 macro_f1=0.8985 bls_acc=0.8387 train=1362 test=341 features=204
```

Inputs to `run_smoke_check(pkl_path, key=None, max_rows=2000, random_state=42)`
are documented in the module; load any pkl by importing it directly.

## Full training CLI: `train.py`

```bash
# single config
uv run --group experiments python experiments/train.py --config experiments/configs/default.yaml
# override config from the CLI
uv run --group experiments python experiments/train.py --dataset biomedical_larger --n-splits 3
# batch: run every *.yaml/*.yml in a directory sequentially
uv run --group experiments python experiments/train.py --config-dir examples/configs --n-splits 5
```

### CLI options

| Option | Overrides | Meaning |
|---|---|---|
| `--config PATH` | — | YAML config path (default `experiments/configs/default.yaml`). |
| `--config-dir DIR` | — | Run every `*.yaml`/`*.yml` in `DIR` (sorted) sequentially. Each config gets its own run dir, JSONL log, and `.npz` side-file. CLI overrides apply to every config in the batch. Conflicts with `--config`. |
| `--dataset NAME` | `dataset` | Dataset stem; loads `{data_path}/{NAME}.pkl`. |
| `--model NAME` | `model.name` | Model. Built-in tier `tbls`/`bls`; **any other name** dispatches to `experiments.classifiers.create_classifier` (see "Baseline models" below). |
| `--map-num N` | `model.map_num` | Legacy alias for `TBLS(n_map_trees=N)` (TBLS only). |
| `--n-splits N` | `cv.n_splits` | Number of `KFold` folds. |
| `--output-dir DIR` | `output_dir` | Where run + cohort outputs are written (see "Output layout"). |
| `--fusion [cca\|gfcca]` | `fusion.method` | Override fusion method for multi-view cohorts (single-view cohorts ignore this). |
| `--grid` | — | Sweep the model hyperparameter grid; see "Grid search" below. Default grid `TBLS_GRID`/`BLS_GRID` in `hyperparams.py`, overridable via YAML `grid:`. Only valid for `tbls`/`bls`; baselines require a YAML `grid:` (warns + falls back to a single k-fold otherwise). |

## YAML config reference

```yaml
dataset: biomedical_larger          # stem of the pkl (loads {data_path}/{dataset}.pkl)
data_path: examples/datasets/       # directory holding the pkl; default experiments/datasets/

run_name: TBLS Full                 # human-chosen experiment name. Becomes the
                                    # run-directory stem AND the JSONL/figure
                                    # label. Optional; defaults to
                                    # {model.name}_{dataset}/{timestamp}.

model:                              # one tier per model.name:
  name: tbls                        #   'tbls'/'bls'  -> in-package estimator with
                                    #     hyperparams from hyperparams.TBLS_DEFAULTS/
                                    #     BLS_DEFAULTS. YAML keys (after the legacy-
                                    #     key translation map_num->n_map_trees,
                                    #     enhance_num->n_enhance_trees) override only
                                    #     valid constructor params.
                                    #   any other name -> create_classifier(name, ...);
                                    #     YAML keys become **kwargs; random_state is read
                                    #     from model.random_state (default 42).
  n_map_trees: 10                   # TBLS: number of mapping trees
  n_enhance_trees: 10               # TBLS: number of enhancement trees
  use_if_weights: true              # TBLS: IFS sample weighting on (its differentiator)
  graph_gamma: 0.1                  # TBLS: graph-Laplacian regularization strength (0=off)
  random_state: 42

preprocess:
  feature_selection: lasso          # lasso | pca | mutual_info | null
  resampling: smote                 # smote | adasyn | border_smote | undersample |
                                    # tomek | smote_tomek | smote_enn | null
                                    # (applied to the train split only, after FS)

cv:
  n_splits: 5                       # number of KFold folds
  random_state: 42

fusion:                             # only relevant for multi-view cohorts (pkl has
  method: gfcca                     #   "views" dict instead of "data"); single-view
  view_groups:                      #   cohorts ignore the whole section.
    - ["view_a", "view_b"]          # See usage-multiview-fusion.md.

output_dir: examples/runs           # run + cohort output goes under here

grid:                               # optional {-axis-name -> list-of-values}.
  use_if_weights: [false, true]     # When present, axes named here REPLACE the same
  graph_gamma: [0.0, 0.05, 0.1]     # named default axis (TBLS_GRID/BLS_GRID). Default
                                    # axes NOT named here are KEPT (override/extend
                                    # semantics). Baselines have no default grid, so a
                                    # YAML `grid:` is required to sweep them.
```

### `feature_selection` parameter internals

| Value | Implementation (in `experiments/dataprocess.py`) |
|---|---|
| `lasso` | `Lasso(alpha=0.01)`; non-zero-coefficient features kept, the mask reused on the test split. |
| `pca` | `PCA(n_components=0.95)`; `transform` applied to both train and test. |
| `mutual_info` | `SelectKBest(mutual_info_classif, k=10)`; `transform` applied in place. |
| `null` / omitted | No feature selection. |

The internal `alpha=0.01` / `n_components=0.95` / `k=10` are not currently
configurable from YAML — if you need to vary them, edit the constants in
`experiments/dataprocess.py::FEATURE_SELECTORS` or call `DataLoader` directly.

### `resampling` parameter internals

| Value | imbalanced-learn class |
|---|---|
| `smote`, `adasyn`, `border_smote` | `SMOTE`, `ADASYN`, `BorderlineSMOTE` (over-sample). |
| `undersample`, `tomek` | `RandomUnderSampler`, `TomekLinks` (under-sample). |
| `smote_tomek`, `smote_enn` | `SMOTETomek`, `SMOTEENN` (combined). |
| `null` / omitted | No resampling. |

All resamplers are applied to the **train split only** after feature
selection; the test split is untouched. `MultiViewDataLoader` (multi-view
cohorts only) rejects SMOTE-family resamplers for `SMOTETomek`/`SMOTEENN` that
need a single-view because of the per-view row-align constraint — see
`usage-multiview-fusion.md`.

## Output layout

For every run, `train.py` writes (relative to `output_dir`, the default fall-back
path `results_dir/`):

```
{output_dir}/{run_name}/{timestamp}/                 <- run dir (loguru + .npz)
    logs/{dataset}_{timestamp}.jsonl                  <- structured log (see below)
    logs/{dataset}_{timestamp}_{cohort}_predictions.npz
                                                     <- raw per-fold predictions
                                                        (non-grid runs only)

{output_dir}/{run_name}/{cohort}/{timestamp}/        <- per-cohort Excel dir
    {cohort}_{model_name}_FS-{...}_RS-{...}.xlsx     <- one xlsx per cohort
```

where `{timestamp}` is `time.strftime("%Y%m%d_%H%M%S")` shared by both
branches (so the `.npz`-carrying run dir and the cohort Excel dir are time-stamp
siblings of `run_name`). The app-level output dir and `examples/runs/`/`plots/`
are git-ignored — same as `dist/`, `.pytest_cache/`, etc.

### Excel sheet layout

The per-cohort xlsx (`{cohort}_{model_name}_FS-..._RS-..._xlsx`) carries, per
`TBLSResultSaver`:

- **Non-grid runs**: `{model}_Details` sheet (one row per fold) +
  `{model}_Summary` sheet (one row, the cross-fold averages under `avg_*`
  key prefixes, plus the cohort key).
- **`--grid` runs**:
  - `Grid_{i:03d}` sheet per grid point (the `_cross_validate` fold rows for that point),
  - `GridSummary` sheet (one row per grid point, ranked by
    `avg_balanced_accuracy` descending; includes the swept hyperparameters,
    every returned metric `avg_*`, plus `rank` and `is_winner` — **row 1 is the
    winner**, `is_winner=True`),
  - `GridSearchLog` sheet (same contents as `GridSummary`, flat pre-sort order,
    so it reads as a chronological search log).

The file's metadata sheet (`{sheet}_Meta`, written by `save_summary`) records
the `Feature_Selection` and `Resampling_Method` used, for traceability.

## Structured JSONL log

`experiments/train.py` calls `experiments.logging_setup.configure_logging` at
run start, which removes loguru's default sink and adds two:

- a human-readable **stdout** sink at level `INFO` (colorized
  `<level>{level: <8}</level> {message}` format), so eyeballed output stays
  simple;
- a structured **JSONL file** sink at level `DEBUG` with
  `serialize=True`, writing one JSON object per line to
  `logs/{dataset}_{timestamp}.jsonl`. Stdlib `logging` is intercepted (an
  `InterceptHandler`) so modules that still use `logging.getLogger` (e.g.
  `experiments.evaluate`'s probability-metric warnings) also land in the JSONL.

### Event schema

Every event line carries loguru's record metadata (`record.text`, `record.level`,
`record.time`, `record.function`, `record.line`, ...) under `"record"`, with
the bound event payload under `record["extra"]`. The typed schemas live in
`experiments/logging_schema.py` (`RunStartedEvent`, `FoldCompletedEvent`,
`GridPointCompletedEvent`, `GridSummaryEvent`, `RunFinishedEvent`) — those
`TypedDict`s are the canonical description; only the discriminators / key
fields are summarized here:

| Event | When | Key fields in `record.extra` |
|---|---|---|
| `run_started` | once per run | `dataset`, `model`, `fusion_method`, `grid`, `run_name` (optional) |
| `fold_completed` | per fold (per cohort) | `dataset`, `cohort_key`, `fold`, `n_splits`, `metrics` (`MetricsDict`), `grid_idx`/`grid_params` (only under `--grid`), `predictions_file` |
| `grid_point_completed` | per swept grid point (under `--grid`) | `dataset`, `cohort_key`, `grid_idx`, `n_grid_points`, `grid_params`, `metrics` (`avg_*`-prefixed averages) |
| `grid_summary` | once per cohort after `--grid` ranks | `dataset`, `cohort_key`, `winner_params`, `winner_metric`, `n_grid_points` |
| `run_finished` | once per run | `dataset`, `duration_seconds` |

The scalar `metrics` dict follows the `MetricsDict` TypedDict in
`experiments/metrics_schema.py` (binary: accuracy/precision/recall/f1/...
plus MCC/Kappa/log_loss/brier_score additively; multiclass: macro/weighted
averages and one-vs-rest specificity/NPV/gmean). The `auroc`/`auprc`/
`optimal_threshold`/`log_loss`/`brier_score` keys degrade to `None` when
`y_score` is missing or fails (e.g. degenerate fold); they are absent for
multiclass except `auroc`.

### `.npz` predictions side-file

To keep the JSONL scalar-only, raw per-fold `y_true`/`y_pred`/`y_score` arrays
are persisted to `logs/{dataset}_{timestamp}_{cohort}_predictions.npz` (one
file per cohort of a non-grid run), keyed by
`{cohort}_fold{N}_{y_true,y_pred,y_score}`. `y_score` is `model.predict_proba(x_te)`
as a `float32` array of shape `(n_test, n_classes)` — **not** a 1-D positive-class
probability; reading code that wants the binary positive class must slice
`y_score[:, 1]` (the visualize/compare CLIs do this). Produced for non-grid
runs only (a 27-point × n_folds × n_cohorts grid would explode the side-file).
The `predictions_file` field in each `fold_completed` event names its side-file
(or is `None` for grid runs / folds that did not persist).

## Hyperparameter defaults and grid search

Defaults and grid axes live in `experiments/hyperparams.py` as plain,
directly-editable Python dicts:

| Dict | Used by | Current value |
|---|---|---|
| `TBLS_DEFAULTS` | single-run defaults merged with any `model:` overrides for `model.name: tbls` | `n_map_trees: 10, n_enhance_trees: 10, tree_max_depth: 5, tree_min_samples_split: 3, tree_max_features_ratio: 0.7, reg_param: 1e-8` (`graph_strategy`/`if_strategy` left at TBLS's constructor defaults `discriminative`/`simple`) |
| `BLS_DEFAULTS` | single-run defaults for `model.name: bls` | `n_feature_groups: 30, n_feature_nodes_per_group: 40, n_enhancement_groups: 1, n_enhancement_nodes_per_group: 500, reg_param: 1.0, map_func/enhance_func: "relu"` |
| `TBLS_GRID` | axes swept by `--grid` for `tbls` (or YAML `grid:` merged with this; see above) | `n_map_trees: [10, 20, 40], n_enhance_trees: [10, 20, 40], reg_param: [1e-8, 1e-4, 1e-2]` (a 3×3×3 = 27-point default) |
| `BLS_GRID` | axes swept by `--grid` for `bls` | `n_feature_groups: [15, 30, 60], n_feature_nodes_per_group: [20, 40, 80], reg_param: [0.1, 1.0, 10.0]` (27 points) |
| `CCA_DEFAULTS`/`CCA_GRID`/`GFCCA_DEFAULTS`/`GFCCA_GRID` | fusion hyperparameters for multi-view cohorts (`CCA_GRID`/`GFCCA_GRID` are NOT swept by `--grid` in this version) | see `usage-multiview-fusion.md` |

### `--grid` semantics (two-tier resolution, see `_resolve_grid` in `train.py`)

1. **Default grid** = `TBLS_GRID` for `model.name: tbls`, `BLS_GRID` for
   `model.name: bls`. To change what the default sweep is, edit the module-level
   dict in `hyperparams.py` (it is intentionally a Python constant, not a YAML
   surface — so the values are version-controlled and reviewed in code review;
   YAML overrides it for a single run).
2. **YAML `grid:` override/extension** — when present, axes named in YAML
   **replace** the same-named default axis exactly (the YAML list wins);
   default axes **not** named in the YAML are **kept**. So `grid:` can shrink
   the default, swap one axis, or add a brand-new axis (e.g.
   `use_if_weights: [false, true]` to sweep a previously-fixed flag).
3. **Baseline models (`lr`/`rf`/...) have no default grid** — a YAML `grid:`
   section is required. `--grid` without any YAML `grid:` for a baseline
   raises a clear `ValueError("No default grid ... Set YAML grid: ...")`.
4. Passing `--grid` for a baseline **without** a YAML `grid:` falls back to a
   single k-fold run with a logged warning (it does **not** silently sweep a
   dummy grid).

### Grid outputs

Per `--grid` cohort:

- `Grid_{i:03d}` sheets (per-point, fold-level CSV rows),
- `GridSummary` (ranked, `is_winner` flag on row 1),
- `GridSearchLog` (flat),
- `grid_point_completed` JSONL events (one per point),
- a `grid_summary` event at the end with `winner_params`/`winner_metric`/
  `n_grid_points`.

`experiments/visualize.py` reads those back and renders the
`grid_search_summary.png` (metric vs. each swept axis, one subplot per axis).

## Baseline models (any model in `experiments.classifiers.create_classifier`)

`model.name: <baseline>` dispatches to
`experiments.classifiers.create_classifier(name, random_state, **kwargs)`.
This is a pre-existing factory built for benchmarking `TBLS`/`BLS` against
standard baselines. Supported names include: `rf`, `svm`, `xgb`, `lgb`,
`catboost`, `knn`, `lr`, `lasso`, `elasticnet`, `nb`, `lda`, `cart`, `mlp`,
`dnn`, `extratrees`, `gbdt`, `block_plsda`, `block_splsda`, `mogonet`,
`mogonet_nn`, `mofa`, `diablo`, `snf` (the full canonical list is in
`experiments/classifiers.py`'s `create_classifier` docstring).

Conventions:

- **Imbalancing**: built-in. Every supported baseline is constructed with
  `class_weight="balanced"` (or its equivalent), so passing `--resampling: null`
  still gives a class-balanced fit.
- **Soft dependencies**: `xgboost` is in the experiments dependency group;
  `lightgbm`, `catboost`, `torch` are **not** (they are imported lazily, so a
  missing optional dep raises a clear `ImportError` from `create_classifier`
  only if you ask for the corresponding `model.name`).
- **YAML `model:` keys become `**kwargs`** to the underlying estimator; only
  `name` and `random_state` are consumed specially. `--grid` is not valid
  without a YAML `grid:` (see above).

## Comparison and visualization CLIs

### `visualize.py — per-fold / grid-search / ROC / PR / confusion plots

Resolves each `--dir` via `experiments/run_resolution.py::resolve_run_dir`
(see "`--dir` resolution rule") and reads the JSONL `fold_completed` +
`grid_point_completed` (+ `grid_summary`) events. Outputs (under
`--output-dir`, default `plots/` next to the first `--dir`):

| File | Source | Scope |
|---|---|---|
| `per_fold_metrics.png` | scalar `fold_completed` metrics | always |
| `grid_search_summary.png` | `grid_point_completed` rows | `--grid` runs only |
| `roc_{cohort}.png` | `.npz` side-file's `y_true`/`y_score` (positive class via `[:, 1]`) | non-grid runs only, **one PNG per cohort** |
| `pr_{cohort}.png` | same | same — **one PNG per cohort** |
| `confusion_{run}.png` | `.npz` side-file's `y_true`/`y_pred` | non-grid runs only, one PNG per run (cohorts as subplots) |

ROC and PR are split per-cohort by design — the meaningful ablation comparison
is "run A vs run B on the same cohort", so each per-cohort file overlays every
run; per-fold bars and the grid-search summary keep their existing layout.

#### `--dir` resolution rule

Applies identically to `visualize.py` and `compare.py`. A `--dir` argument is
either:

- the **run-name layer** (e.g. `examples/runs/TBLS Full`) — the CLI auto-picks
  the newest `YYYYMMDD_HHMMSS` timestamp subdirectory under it; or
- the **run-name/<timestamp>** layer (e.g.
  `examples/runs/TBLS Full/20260724_074140`) — used directly.

Anything deeper (e.g. `.../<timestamp>/logs`), shallower (e.g. `examples/runs`),
or whose subdirectory is not a `YYYYMMDD_HHMMSS` name **errors out** with a
clear diagnostic — there is no shell-globbing, no silent stale-run pick. Run
names may contain spaces (Sheet names, paths, and legend labels all keep the
space). `compare.py` additionally uses `run_resolution.cohort_excel_dir` to
find the sibling per-cohort Excel dir under the **same** timestamp — it refuses
a timestamp mismatch with the same error.

#### `visualize.py` CLI options

| Option | Default | Meaning |
|---|---|---|
| `--dir DIR [`--dir DIR ...`]` | — | One or more run directories (resolved as above). |
| `--output-dir DIR` | `plots/` next to the first `--dir` | Where PNGs are written. |
| `--dpi N` | `300` | PNG resolution. Lower with `--dpi 120` for quick previews. |

### `compare.py` — cross-run comparison Excel

Resolves each `--dir` (same rule), parses every `fold_completed` event across
runs, and writes `comparison.xlsx` under `--output-dir` (default
`examples/comparison`):

- **one sheet per cohort** + a `README` sheet documenting the layout.
- Rows = runs (one per `--dir`), sorted; columns = 15 scalar metrics in
  `ORDERED_METRICS` order (`balanced_accuracy`, `accuracy`, `f1_score`, `mcc`,
  `cohen_kappa`, `auroc`, `auprc`, `recall`, `specificity`, `precision`,
  `negative_predictive_value`, `gmean`, `hamming_loss`, `log_loss`,
  `brier_score`).
- Each cell = `mean (std)` across CV folds. Add `--no-std` for bare means.
- **Bold** = best run for that metric on that cohort, chosen by direction
  (`METRIC_DIRECTION`: higher-is-better for `auroc`/`balanced_accuracy`/`mcc`/
  `cohen_kappa`/`accuracy`/`f1_score`/`recall`/`specificity`/`precision`/
  `negative_predictive_value`/`gmean`/`auprc`; lower-is-better for
  `hamming_loss`/`log_loss`/`brier_score`).
- A run that did not produce a cohort leaves that cell blank (not 0, not NaN).

#### `compare.py` CLI options

| Option | Default | Meaning |
|---|---|---|
| `--dir DIR [`--dir DIR ...`]` | — | One or more run directories (resolved via `resolve_run_dir`). |
| `--output-dir DIR` | `examples/comparison` | Where `comparison.xlsx` is written. |
| `--no-std` | off | Drop the `(std)` term; write bare means instead of `mean (std)`. |

## Reading the figures (PR curve cliff on uncalibrated TBLS outputs)

TBLS (and `TBLS Full`/`TBLS Graph`/`TBLS IFS`) produces a `predict_proba` that
is the ridge-regression output `Z = W A_{enh}` softmax-transformed, **without
any calibration step** (no Platt scaling, no sigmoid/isotonic calibration). On
some datasets this leaves a large chunk of low-confidence samples with scores
clustered tightly around `0.5` (the `/2` point of the softmax), so the PR-curve
threshold sweep crosses a "score-density plateau" — when the threshold drops
below this plateau a whole block of samples becomes "predicted positive"
simultaneously, and only ~`prevalence` of them really are positive, so
precision collapses to `prevalence` and recall jumps sharply. This looks like a
near-vertical cliff on the PR curve (see `docs/usage-figures-and-calibration.md`
for the full mathematics and reproducer, a TBLS-vs-LR visualization of the
phenomenon, and the recommended mitigations — Platt scaling or a grouped-bin
calibration as a future work-item, not a current implementation bug).

LR's probabilities come from the convex logistic-loss optimum directly, so
they gradient smoothly across `[0, 1]` without a `0.5` plateau, which is why
the LR PR curves in the same figures look smooth and the TBLS ones don't.
**ROC curves are less affected** because changing the threshold distribution
changes TPR/FPR more smoothly (the ROC integral is invariant to monotone
score-reshaping).

## `--grid` for multi-view cohorts

`--grid` sweeps only the model grid (`TBLS_GRID`/`BLS_GRID` or YAML `grid:`).
Fusion hyperparameters (`CCA_GRID`/`GFCCA_GRID` in `hyperparams.py`) are NOT
swept by `--grid` in this version — documented scope limit, not silently
dropped. See [`usage-multiview-fusion.md`](./usage-multiview-fusion.md) for the
multi-view pkl contract and the per-view row-alignment/resampling restrictions
that motivate this scope decision.

## Smoke / sanity commands summary

| Want | Run |
|---|---|
| One-line dataset+model sanity | `uv run --group experiments python experiments/smoke_run.py` |
| Single TBLS run from a config | `uv run --group experiments python experiments/train.py --config examples/configs/tbls_full.yaml` |
| Batch run every config in a directory | `uv run --group experiments python experiments/train.py --config-dir examples/configs --n-splits 5` |
| Ablation figures overlay | `uv run --group experiments python experiments/visualize.py --dir "examples/runs/TBLS" --dir "examples/runs/TBLS Full" ...` |
| Comparison Excel (mean (std), bold best) | `uv run --group experiments python experiments/compare.py --dir "examples/runs/TBLS" ...` |
| Grid search sweep | `uv run --group experiments python experiments/train.py --config examples/configs/tbls_grid.yaml --grid` |