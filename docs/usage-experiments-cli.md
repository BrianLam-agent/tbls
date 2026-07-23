English | [简体中文](./usage-experiments-cli.zh-CN.md)

# Running the experiments CLI (training on real datasets)

`experiments/` is the training/evaluation pipeline used to run `tbls`
estimators against real datasets on your own machine. It is **not** part of
the published `tbls` package (it depends on `pandas`, `imbalanced-learn`,
`xgboost`, `typer`, `pyyaml`, `openpyxl` — heavy, opinionated dependencies
that would otherwise weigh down every `pip install tbls` user). See
[`architecture.md`](./architecture.md#3-why-the-packageexperiments-split) for
the rationale.

## Setup

```bash
git clone https://github.com/BrianLam-agent/tbls.git
cd tbls
uv sync --group dev --group experiments
```

Place your dataset `.pkl` files under `experiments/datasets/` (this directory
is git-ignored — see [`experiments/datasets/README.md`](../experiments/datasets/README.md)).
The expected pkl shape is either:

- a flat `{"data": X, "target": y}` dict, or
- a multi-key dict of such sub-dataset dicts (each value processed
  independently, keyed by its dict key — e.g. one file holding several
  disease cohorts).

Samples with label `-1` are dropped; labels are binarized to `{0, 1}`
(`(y > 0).astype(int)`), matching the legacy pipeline's convention.
Feature matrices stored as `dtype=object` are coerced to `float64`, and
`NaN`/`Inf` values are zeroed.

## Minimal sanity check: `smoke_run.py`

Before running the full CLI, `experiments/smoke_run.py` is the fastest way to
confirm a dataset loads correctly and `TBLS` fits and predicts sanely on it
(small model, one train/test split, a handful of assertions — finite
`predict_proba`, rows summing to 1, non-degenerate predictions):

```bash
uv run --group experiments python experiments/smoke_run.py
```

```
TBLS smoke check OK | key=DM acc=0.9208 macro_f1=0.8985 bls_acc=0.8387 train=1362 test=341 features=204
```

By default this loads `experiments/datasets/biomedical_larger.pkl`. Edit the
`pkl_path` in `smoke_run.py::main()` (or import `run_smoke_check` directly)
to point at a different file:

```python
from pathlib import Path
from experiments.smoke_run import run_smoke_check

result = run_smoke_check(Path("experiments/datasets/data_cross_train.pkl"), max_rows=2000)
print(result)
```

## Full training CLI: `train.py`

```bash
uv run --group experiments python experiments/train.py
uv run --group experiments python experiments/train.py --dataset biomedical_larger --n-splits 3
uv run --group experiments python experiments/train.py --config experiments/configs/default.yaml --map-num 20
```

| Option | Overrides config key | Meaning |
|---|---|---|
| `--config PATH` | — | YAML config path (default `experiments/configs/default.yaml`). |
| `--dataset NAME` | `dataset` | Dataset stem; loads `{data_path}/{NAME}.pkl`. |
| `--map-num N` | `model.map_num` | `TBLS(n_map_trees=N)`. |
| `--n-splits N` | `cv.n_splits` | Number of `KFold` folds. |
| `--output-dir DIR` | `output_dir` | Where Excel results are written. |

`experiments/configs/default.yaml`:

```yaml
dataset: biomedical_larger
data_path: experiments/datasets/

model:
  name: tbls
  map_num: 10
  enhance_num: 10
  reg_param: 2.0e-15

preprocess:
  feature_selection: lasso   # lasso | pca | mutual_info | null
  resampling: smote          # smote | adasyn | border_smote | undersample |
                              # tomek | smote_tomek | smote_enn | null

cv:
  n_splits: 5
  random_state: 42

output_dir: results_dir
```

### What `train.py` does, per sub-dataset key

1. Loads the pkl (all sub-dataset keys, or the single flat dict under key
   `"single"`).
2. For each key, runs `sklearn.model_selection.KFold` (shuffled,
   `cv.random_state`).
3. Per fold: fits `experiments.dataprocess.DataLoader`'s feature-selection +
   resampling **on the training split only** (no leakage into the test
   split), fits `tbls.TBLS`, evaluates with
   `experiments.evaluate.TBLSEvaluator.calculate_metrics` (accuracy,
   precision, recall, F1, specificity, balanced accuracy, g-mean, AUROC,
   AUPRC, optimal threshold).
4. Writes per-fold results and the cross-fold average to
   `{output_dir}/tbls_{dataset}/{key}/{timestamp}/{key}_tbls_FS-{...}_RS-{...}.xlsx`
   via `experiments.evaluate.TBLSResultSaver`.

Example log output (four sub-datasets in one pkl, 2-fold CV):

```
INFO dataset=biomedical_larger keys=['DM', 'CKD', 'BC', 'CG']
INFO === biomedical_larger / DM : X=(1703, 204) y=(1703,) ===
INFO dataset=biomedical_larger key=DM fold=1/2 acc=0.9085
INFO dataset=biomedical_larger key=DM fold=2/2 acc=0.9166
INFO dataset=biomedical_larger key=DM avg={'avg_accuracy': 0.9125, ...}
```

## Feature selection and resampling options

From `experiments/dataprocess.py::DataLoader`:

| `feature_selection` | Implementation |
|---|---|
| `lasso` | `sklearn.linear_model.Lasso(alpha=0.01)`; non-zero coefficients selected, mask reused on the test split. |
| `pca` | `sklearn.decomposition.PCA(n_components=0.95)`. |
| `mutual_info` | `SelectKBest(mutual_info_classif, k=10)`. |
| `null` / omitted | No feature selection. |

| `resampling` | Implementation |
|---|---|
| `smote`, `adasyn`, `border_smote` | Over-sampling (`imblearn.over_sampling`). |
| `undersample`, `tomek` | Under-sampling (`imblearn.under_sampling`). |
| `smote_tomek`, `smote_enn` | Combined (`imblearn.combine`). |
| `null` / omitted | No resampling. |

Resampling is applied to the training split only, after feature selection.

## Comparison classifiers

`experiments/classifiers.py` is a large factory (`rf`, `svm`, `xgb`, `knn`,
`lr`, `cart`, `mlp`, `extratrees`, `gbdt`, `bls`, `tbls`, and more — see the
module docstring for the full list) used to benchmark `TBLS`/`BroadLearningSystem`
against standard baselines. Optional dependencies it references
(`lightgbm`, `catboost`, `torch`, `muon`) are soft — each is guarded by its
own `try/except ImportError`, so the factory degrades gracefully if you have
not installed them; `uv sync --group experiments` only installs the
dependencies `tbls`'s own training pipeline needs (`xgboost` included,
`lightgbm`/`catboost`/`torch` not).

## Where results go

`results_dir/` (or whatever `output_dir` you configure) is git-ignored — see
the root `.gitignore`. Nothing under it is meant to be committed; treat it as
scratch output, same as `dist/`, `.pytest_cache/`, etc.
