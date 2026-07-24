# The experiments pipeline — overview

`experiments/` is the project's training, evaluation, and comparison pipeline.
It runs the `tbls` estimator (and baselines) on real datasets, dumps structured
logs + Excel + figures, and side-by-side compares multiple runs. It is **not**
part of the published `tbls` package (heavier dependencies), so users install
it with `uv sync --group experiments`.

This `docs/experiments/` directory is the only documentation surface for the
pipeline. It is split so a reader can either skim the
[5-step quick start](#5-step-quick-start) below, or jump to a per-topic page
when they need detail on one thing.

## Pages

| Want | Read |
|---|---|
| Put a dataset on disk and run one experiment | [datasets.md](datasets.md) |
| Write the YAML for one run (every key, every value) | [config-reference.md](config-reference.md) |
| Pick a model name (`tbls`/`bls`/`lr`/`rf`/...) | [models.md](models.md) |
| Run `train.py` (every `--option`, what it does, conflicts) | [cli-train.md](cli-train.md) |
| Make figures from one or more runs | [cli-visualize.md](cli-visualize.md) |
| Make a cross-run comparison Excel | [cli-compare.md](cli-compare.md) |
| Sweep hyperparameters with `--grid` | [grid-search.md](grid-search.md) |
| Understand what's on disk after a run (Excel sheets, JSONL events, npz) | [outputs.md](outputs.md) |
| Why a PR plot looks like it has a cliff | [figures-and-calibration.md](figures-and-calibration.md) |
| Change/extend the pipeline itself | [internals.md](internals.md) |

## 5-step quick start

Assumes you have the repo cloned and a dataset that follows the
[pkl contract](datasets.md). All commands run from the repo root.

### 1. Install the experiments environment

```bash
uv sync --group experiments
```

That installs everything heavy (`pandas`, `imbalanced-learn`, `openpyxl`,
`loguru`, `matplotlib`, `typer`, `pyyaml`, `xgboost`) the pipeline uses.

### 2. Put a dataset where configs expect it

```bash
cp experiments/datasets/biomedical_larger.pkl examples/datasets/
```

The example configs read `examples/datasets/{dataset}.pkl`; the dataset
directory is git-ignored (massive binary). What your pkl must contain:
[datasets.md](datasets.md).

### 3. Pick or write a config — start from an example

Ready-made ablation configs live in `examples/configs/`:

```bash
ls examples/configs/
# tbls_plain.yaml  tbls_ifs.yaml  tlbs_graph.yaml  tbls_full.yaml  lr_baseline.yaml
```

Each is a one-run YAML; open one, change `dataset:` / `model.name:` /
`preprocess:` to taste (every key documented in
[config-reference.md](config-reference.md)). The minimal one is:

```yaml
dataset: biomedical_larger
data_path: examples/datasets/
run_name: My run
model: {name: tbls, use_if_weights: true, graph_gamma: 0.1}
preprocess: {feature_selection: lasso, resampling: smote}
cv: {n_splits: 5, random_state: 42}
output_dir: examples/runs
```

`run_name` becomes both the run-directory stem and the figure legend label, so
it is what you write, not an auto-generated `tbls_biomedical_larger/timestamp`
slug.

### 4. Run it (single config, or batch a whole directory)

```bash
# One config
uv run --group experiments python experiments/train.py \
    --config examples/configs/tbls_full.yaml

# Every *.yaml in a directory, with the same --n-splits override
uv run --group experiments python experiments/train.py \
    --config-dir examples/configs --n-splits 2
```

What lands on disk: a run directory + per-cohort Excel + JSONL + (for non-grid
runs) raw per-fold predictions. Full layout in [outputs.md](outputs.md).

### 5. Make figures and the comparison Excel

```bash
# Figures — give the run-name directory; the CLI finds the newest tingestamp
uv run --group experiments python experiments/visualize.py \
    --dir "examples/runs/TBLS Full" \
    --dir "examples/runs/Logistic Regression" \
    --output-dir examples/plots --dpi 300

# Mean ± std comparison Excel with the best value per metric bolded
uv run --group experiments python experiments/compare.py \
    --dir "examples/runs/TBLS Full" \
    --dir "examples/runs/Logistic Regression" \
    --output-dir examples/comparison
```

`--dir` can take either the run-name layer (auto-picks the newest
`YYYYMMDD_HHMMSS` timestamp subdirectory) or the run-name/timestamp layer
(used directly). Anything deeper, shallower, or non-timestamp errors out — no
shell globbing needed. Full rules in [cli-visualize.md](cli-visualize.md) and
[cli-compare.md](cli-compare.md).

## What is NOT in this scope

- The `tbls`/`BroadLearningSystem` estimator API itself: that is
  [`../usage-tbls.md`](../usage-tbls.md) and [`../usage-bls.md`](../usage-bls.md).
- Multi-view fusion (pkl + CCA/GFCCA): that is
  [`../usage-multiview-fusion.md`](../usage-multiview-fusion.md), and the
  `preprocess` / `fusion` config keys here just feed it.
- Why PR curves look pathological on uncalibrated TBLS outputs:
  [`figures-and-calibration.md`](figures-and-calibration.md).