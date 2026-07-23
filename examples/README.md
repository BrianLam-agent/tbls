# Examples — runnable, end to end

This directory is the quickest way from "I just cloned the repo" to "I have a
fitted TBLS, a baseline comparison, and plots on disk." It holds two kinds of
runnable examples:

- **`examples/01_train_tbls.py`** and **`examples/02_train_tbls_with_grid_search.py`** —
  short, heavily-commented Python scripts that call the `tbls`/`experiments`
  APIs directly. Good for reading one file top to bottom.
- **`examples/configs/*.yaml`** — YAML experiment configs that drive
  `experiments/train.py` via its CLI (no Python). Good for "configure one,
  configure many, run them all in one go". This is how the **TBLS ablation
  study** below is run.

Everything in `experiments/` is installed via the `experiments` uv dependency
group (heavier than the published package: `pandas`, `imbalanced-learn`,
`openpyxl`, `loguru`, `matplotlib`, ...). The `examples/` directory is **not**
part of the published wheel — like `experiments/`, it only ships in the source
repo.

---

## Step 0 — set up the environment

From the repo root:

```bash
uv sync --group experiments
```

This installs every dependency `experiments/train.py`, `experiments/visualize.py`,
and the example scripts need. You only do this once.

---

## Step 1 — put a dataset where the configs expect it

The configs all set `data_path: examples/datasets/` and load the
`biomedical_larger` dataset, so they read
`examples/datasets/biomedical_larger.pkl`. That `.pkl` is **git-ignored**
(large binary; see `.gitignore`) — you must provide it locally.

Copy it from the canonical location (`experiments/datasets/`, where the test
suite reads it from):

```bash
cp experiments/datasets/biomedical_larger.pkl examples/datasets/
```

(You would need the file there anyway if you ran the test suite or the
training CLI against the real dataset — this is just the examples' preferred
location.)

**Expected pkl format:** `joblib.load(...)` returns either a flat `{"data": X,
"target": y}` dict or a multi-key dict of such sub-datasets (the example
dataset `biomedical_larger.pkl` is multi-key with cohorts `DM`, `CKD`, `BC`,
`CG`; each config runs on all of them). Samples with label `-1` are dropped;
labels are binarized to `{0, 1}`. See
[`examples/datasets/README.md`](./datasets/README.md) and
[`experiments/datasets/README.md`](../experiments/datasets/README.md) for
details.

---

## Step 2 — pick a config and run it

The five configs in `examples/configs/` are one self-contained TBLS ablation
study plus a logistic-regression baseline:

| Config | `model.name` | `use_if_weights` | `graph_gamma` | What it is |
|---|---|---|---|---|
| `tbls_plain.yaml` | `tbls` | false | 0.0 | Plain TBLS (no IFS, no graph) |
| `tbls_ifs.yaml`   | `tbls` | true  | 0.0 | FTBLS — IFS sample weighting only |
| `tbls_graph.yaml` | `tbls` | false | 0.1 | TBLS with graph-Laplacian only |
| `tbls_full.yaml`  | `tbls` | true  | 0.1 | GFTBLS — both regularizers on (the fully-regularized TBLS) |
| `lr_baseline.yaml`| `lr`   | —     | —     | Logistic Regression baseline |

Every config sets `run_name:` (so the run directory and the figures say
`tbls_full`, not `tbls_biomedical_larger/20260724_074140`) and
`output_dir: examples/runs`.

### Run one config

```bash
uv run --group experiments python experiments/train.py \
    --config examples/configs/tbls_full.yaml
```

This runs 2-fold CV on every cohort (`DM`, `CKD`, `BC`, `CG`) and writes, under
`examples/runs/tbls_full/<timestamp>/`:

- `logs/biomedical_larger_<timestamp>.jsonl` — structured per-fold events
  (one JSON object per line);
- `logs/biomedical_larger_<timestamp>_<cohort>_predictions.npz` — raw
  per-fold `y_true`/`y_pred`/`y_score` (one file per cohort, for ROC/PR/
  confusion-matrix plots);
- per-cohort Excel files beside the run dir.

### Run all five at once — `--config-dir`

```bash
uv run --group experiments python experiments/train.py \
    --config-dir examples/configs --n-splits 2
```

`--config-dir` runs every `*.yaml`/`*.yml` in the directory (sorted by
filename) sequentially. CLI overrides (`--n-splits`, `--dataset`, ...) apply
to every config in the batch. Each config gets its own run directory under
`examples/runs/<run_name>/<timestamp>/`.

---

## Step 3 — visualize all runs on one set of figures

After the five runs above, point `experiments/visualize.py` at the five run
directories (their timestamp sub-directory, which contains the `logs/`):

```bash
uv run --group experiments python experiments/visualize.py \
    --dir examples/runs/tbls_plain/<timestamp> \
    --dir examples/runs/tbls_ifs/<timestamp> \
    --dir examples/runs/tbls_graph/<timestamp> \
    --dir examples/runs/tbls_full/<timestamp> \
    --dir examples/runs/lr_baseline/<timestamp> \
    --output-dir examples/plots --dpi 300
```

Replace each `<timestamp>` with the actual `20260724_…` directory the runs
created (or use a shell glob: `--dir examples/runs/tbls_full/2*`). Output in
`examples/plots/`:

| File(s) | What you see |
|---|---|
| `per_fold_metrics.png` | Per-cohort `balanced_accuracy`/`mcc` bars, grouped by run — the headline ablation comparison. |
| `roc_<cohort>.png` × 4 | One ROC file per cohort (`roc_DM.png`, …); each file overlays the 5 runs so you read "on this cohort, which run is best?". |
| `pr_<cohort>.png` × 4   | Same split, precision-recall. |
| `confusion_<run>.png` × 5 | One confusion-matrix sheet per run (cohorts as sub-plots). |
| `grid_search_summary.png` | only if any `--grid` run was passed; the five configs above are not `--grid`, so this is skipped. |

ROC/PR are **split per cohort** by design: the meaningful ablation
comparison is "run A vs run B on the *same* cohort", not "run A's curves across
cohorts". Cranking them into one giant legend defeats the purpose.

`--dpi 300` gives print-quality PNGs (the default; lower it with `--dpi 120`
for quick previews).

---

## The two Python-script examples (one-file reads)

If you'd rather read one Python file top-to-bottom than drive a CLI:

### `examples/01_train_tbls.py` — a single TBLS training run

```bash
uv run --group experiments python examples/01_train_tbls.py
```

Loads `"DM"`, a single stratified train/test split (no CV), Lasso-selects
features fit on the train split only, fits one `TBLS(use_if_weights=True)`,
prints held-out `accuracy`/`balanced_accuracy`/`macro_f1`.

Expected output (approximate; varies slightly across machines and BLAS):

```text
Worked example 01: single TBLS run (cohort=DM)
  train=1362 test=341 features_in=204 features_selected=62
  accuracy          = 0.9179
  balanced_accuracy = 0.9099
  macro_f1          = 0.8943
```

### `examples/02_train_tbls_with_grid_search.py` — a programmatic `--grid` sweep

```bash
uv run --group experiments python examples/02_train_tbls_with_grid_search.py
```

Drives `experiments.train._run_grid` programmatically (the same internal the
test suite calls — no subprocess), runs a small 2×2 grid (`n_map_trees ∈
{10,20}`, `reg_param ∈ {1e-8,1e-4}`) with 2-fold CV on `"DM"`, and prints the
ranked `GridSummary` rows (sorted by `avg_balanced_accuracy` descending).
Excel sheets go to a throwaway temp dir; the printed rows are exactly the
`GridSummary` sheet contents:

```text
Worked example 02: TBLS grid search (cohort=DM, grid=2x2, 2-fold CV)
  rank grid_idx n_map_trees  reg_param  avg_acc avg_bal_acc   avg_f1 avg_auroc
  ----------------------------------------------------------------------------
     1        1          10      1e-08   0.9184      0.9026   0.8412    0.8790
     ...
```

---

##yaml reference (one config, annotated)

Every config has the same shape; here is `tbls_full.yaml` annotated:

```yaml
dataset: biomedical_larger          # stem of the pkl (loads {data_path}/{dataset}.pkl)
data_path: examples/datasets/       # directory holding the pkl

run_name: tbls_full                 # human-chosen experiment name -- becomes the
                                    # run-directory stem AND the figure label

model:
  name: tbls                        # 'tbls' / 'bls' build the in-package estimator
                                    # (with hyperparams from experiments/hyperparams.py);
                                    # any other name dispatches to
                                    # experiments.classifiers.create_classifier
                                    # (lr, rf, svm, knn, xgb, ..., see that module).
  n_map_trees: 10                   # number of mapping (random-feature) trees
  n_enhance_trees: 10               # number of enhancement trees
  use_if_weights: true              # IFS sample weighting on (the TBLS differentiator)
  graph_gamma: 0.1                  # graph-Laplacian regularization strength (0 = off)
  random_state: 42

preprocess:
  feature_selection: lasso          # lasso | pca | mutual_info | null
  resampling: smote                 # smote | adasyn | border_smote | undersample |
                                    # tomek | smote_tomek | smote_enn | null
                                    # (applied to the train split only, after selection)

cv:
  n_splits: 2                       # number of KFold folds
  random_state: 42

output_dir: examples/runs           # run + cohort output goes under here
```

`run_name` is optional. If you omit it, the run directory falls back to
`{model.name}_{dataset}/{timestamp}` and the figure label to that path's last
component — that is the older behavior; setting `run_name` is recommended for
any comparison you want to label cleanly.

---

## Adding your own dataset / model

- **Different dataset that already follows the pkl contract:** copy its `.pkl`
  into `examples/datasets/`, set `dataset:` and `data_path:` in a new YAML in
  `examples/configs/`, set a distinct `run_name:`, and re-run Step 2/3.
- **A baseline model:** set `model.name:` to any of `create_classifier`'s
  identifiers (`rf`, `svm`, `knn`, `lr`, `xgb`, `lgb`, `catboost`, `cart`,
  `mlp`, `extratrees`, `gbdt`, `nb`, `lda`, ...). YAML `model:` overrides
  become `**kwargs` forwarded to the constructor; `random_state` is read from
  `model.random_state` (default 42). The soft deps (`xgboost`, `lightgbm`,
  `catboost`, `torch`) raise a clear `ImportError` if not installed.
- **`--grid` only sweeps `tbls`/`bls`** (the in-package estimators) via
  `TBLS_GRID`/`BLS_GRID` in `experiments/hyperparams.py`. Passing `--grid`
  with a baseline falls back to a single k-fold run and warns — it is not
  silently dropped.

---

## Full API / CLI reference

These examples do not repeat the documented API/CLI surface:

- [`docs/usage-tbls.md`](../docs/usage-tbls.md) — the full `TBLS` tutorial
  (parameters, IFS / graph-Laplacian regularization, incremental layers,
  reproducibility, performance notes).
- [`docs/usage-bls.md`](../docs/usage-bls.md) — `BroadLearningSystem` tutorial.
- [`docs/usage-cca-gfcca.md`](../docs/usage-cca-gfcca.md) — `PairwiseKCCA` /
  `GraphFuzzyKCCA` two-view feature extractors.
- [`docs/usage-experiments-cli.md`](../docs/usage-experiments-cli.md) — the
  full training-CLI reference (every `--option`, the YAML config, the JSONL
  event schema, the `.npz` predictions side-file, multi-view `--fusion`).
- [`docs/usage-multiview-fusion.md`](../docs/usage-multiview-fusion.md) —
  multi-view pkl data contract + CCA/GFCCA fusion-group config.

For the minimal runnable precedent these examples mirror, see
[`experiments/smoke_run.py`](../experiments/smoke_run.py).