English | [简体中文](./config-reference.zh-CN.md)

# YAML config reference

A YAML config drives one run of `experiments/train.py`. This page lists every
key, what it does, what value it expects, what happens if you leave it out,
and a minimal example. For *which `model.name` values exist*, see
[models.md](models.md); for the CLI options that override these, see
[cli-train.md](cli-train.md).

## Top-level keys

| Key | Required? | Type | Default | Purpose |
|---|---|---|---|---|
| `dataset` | yes | string | — | Dataset stem; loads `{data_path}/{dataset}.pkl`. |
| `data_path` | no | string | `experiments/datasets` | Directory holding the pkl. |
| `run_name` | no | string | `{model.name}_{dataset}` | Becomes the run-directory stem AND the figure/Excel label. Set it for clean labels; spaces are fine. |
| `model` | yes | mapping | `{name: tbls}` | Pick the model + its constructor kwargs. |
| `preprocess` | no | mapping | `{}` | Feature selection + resampling applied to the train split only. |
| `cv` | no | mapping | `{n_splits: 5, random_state: 42}` | Cross-validation fold count and seed. |
| `fusion` | no | mapping | — | Only for multi-view cohorts; ignored for single-view pkl. See [../usage-multiview-fusion.md](../usage-multiview-fusion.md). |
| `grid` | no | mapping | (default `TBLS_GRID`/`BLS_GRID`) | Hyperparameter axes to sweep under `--grid`. See [grid-search.md](grid-search.md). |
| `output_dir` | no | string | `results_dir` | Where the run dir + per-cohort Excel are written. |

## `model` (pick the estimator)

| Sub-key | Required? | Default | Effect |
|---|---|---|---|
| `name` | yes | `tbls` | Which estimator. `tbls`/`bls` are the package's own; anything else must be a supported baseline name from [models.md](models.md). Unknown names raise a `ValueError` naming the supported set. |
| any other key | no | — | Forwarded as a constructor kwarg to the estimator (`tbls`/`bls` respect the package defaults from `hyperparams.py`; baselines accept whatever their underlying sklearn class accepts). |
| `random_state` | no | `42` | Seed; applies to `tbls`/`bls` and all baselines (the base `random_state` is read here for both tiers). |

Legacy keys `map_num`/`enhance_num` are aliased to `n_map_trees`/`n_enhance_trees` for compatibility with older YAML configs.

Example:

```yaml
model:
  name: tbls
  n_map_trees: 10
  n_enhance_trees: 10
  use_if_weights: true
  graph_gamma: 0.1
  random_state: 42
```

## `preprocess` (feature selection + resampling, train split only)

| Sub-key | Effect | Possible values | Default |
|---|---|---|---|
| `feature_selection` | Pick a feature selector fit on train, reused for test. | `lasso`, `pca`, `mutual_info`, `null` | `null` (no selection) |
| `resampling` | Pick an imbalanced-learn sampler applied to the train split AFTER feature selection. The test split is never resampled. | `smote`, `adasyn`, `border_smote`, `undersample`, `tomek`, `smote_tomek`, `smote_enn`, `null` | `null` (no resampling) |

Internal fixed hyperparameters of the selectors (not currently configurable
from YAML — edit `experiments/dataprocess.py` if you need to vary them):

| `feature_selection` | Implementation |
|---|---|
| `lasso` | `Lasso(alpha=0.01)`; non-zero coefficients kept. |
| `pca` | `PCA(n_components=0.95)`. |
| `mutual_info` | `SelectKBest(mutual_info_classif, k=10)`. |

Resamplers map directly to the imbalanced-learn classes of the same name.

## `cv` (cross-validation)

| Sub-key | Effect | Possible values | Default |
|---|---|---|---|
| `n_splits` | Number of `KFold` folds (shuffled). | integer ≥ 2 | `5` |
| `random_state` | Seed for `KFold(shuffle=True)`. | integer | `42` |

## `output_dir`

A relative (from repo root) or absolute directory. The CLI creates, under it:

```
{output_dir}/{run_name}/{timestamp}/                 <- run dir (logs + npz)
{output_dir}/{run_name}/{cohort}/{timestamp}/        <- per-cohort Excel
```

Full layout in [outputs.md](outputs.md).

## `run_name`

Optional but recommended. With it set, your dir is
`examples/runs/TBLS Full/` and the figure legend reads `TBLS Full`. Without
it, you get the fallback `examples/runs/tbls_biomedical_larger/<timestamp>/`
and the legend is that auto-path. Spaces are fine (Windows + openpyxl + log
paths all accept them).

## `grid` (only matters with `--grid`)

A mapping of `{axis-name: [list of values]}`. When you pass `--grid`:

- `model.name: tbls` → default axes swept are `n_map_trees` / `n_enhance_trees` / `reg_param`; anything you put in `grid:` *replaces* the matching default axis, and any default axis you don't mention is *kept*.
- `model.name: bls` → same merge against `BLS_GRID`.
- baselines → no default grid; `grid:` is *required* for `--grid`.

Full semantics: [grid-search.md](grid-search.md).

## Full annotated example

```yaml
# An ablation run: completely-regularized TBLS (GFTBLS)
dataset: biomedical_larger
data_path: examples/datasets/
run_name: TBLS Full

model:
  name: tbls
  n_map_trees: 10
  n_enhance_trees: 10
  use_if_weights: true
  graph_gamma: 0.1
  random_state: 42

preprocess:
  feature_selection: lasso
  resampling: smote

cv:
  n_splits: 5
  random_state: 42

output_dir: examples/runs
```