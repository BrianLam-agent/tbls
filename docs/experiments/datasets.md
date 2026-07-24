English | [简体中文](./datasets.zh-CN.md)

# Datasets

`experiments/train.py` loads datasets through `experiments.dataprocess.py::
DataLoader`. Put the dataset file under your `data_path` directory (the example
configs use `examples/datasets/`; `experiments/configs/default.yaml` uses
`experiments/datasets/`). Dataset directories are git-ignored (large binaries),
so you place them locally — they are never committed.

## Pickle format (the only one you should use)

`joblib.load(...)` on a `.pkl` returns a dict. `DataLoader` accepts one of:

### 1. Flat single-cohort
```python
{"data": X, "target": y}
```
- `X` shape `(n, d)`, `y` shape `(n,)`.
- Reported under key `"single"`.

### 2. Multi-key (several cohorts in one pkl)
```python
{"DM": {"data": X_DM, "target": y_DM},
 "CKD": {"data": X_CKD, "target": y_CKD},
 ...}
```
- Each value processed independently, keyed by its dict key.
- `train.py` iterates every cohort; outputs are written per-cohort under that
  key (`{cohort}/{timestamp}/...`).

### 3. Multi-view (CCA/GFCCA fusion)
```python
{"views": {"view_a": X_a, "view_b": X_b, ...}, "target": y}
```
- Auto-detected (a `"views"` key instead of `"data"`).
- Needs a `fusion:` YAML block to configure `method`/`view_groups`; see
  [../usage-multiview-fusion.md](../usage-multiview-fusion.md). Single-view
  YAMLs ignore fusion entirely.

## Canonical preprocessing applied by `DataLoader`

(Already baked in; you don't need to do it.)

- Label `-1` samples are dropped.
- Labels are binarized to `{0, 1}` via `(y > 0).astype(int)` — this pipeline
  is a binary classification pipeline.
- `dtype=object` feature matrices are coerced to `float64`.
- `NaN` and `Inf` feature values are zeroed.

## Where to put the file

`examples/configs/*.yaml` set `data_path: examples/datasets/`. The canonical
copy lives in `experiments/datasets/` (used by the test suite + smoke run);
copy it to the example location:

```bash
cp experiments/datasets/biomedical_larger.pkl examples/datasets/
```

See [`experiments/datasets/README.md`](../../experiments/datasets/README.md)
for the canonical files present on this dev machine.

## Legacy CSV+label pair (do NOT use for binary TBLS experiments)

`DataLoader` also has a `_load_csv` path that fires when both
`{dataset}_data.csv` and `{dataset}_label.csv` exist (it tries CSV before pkl).
This path was kept for a legacy multi-label workflow and uses
`MultiLabelBinarizer` — **it does not binarize to {0, 1}**, does not drop
label `-1`, and is not the path the TBLS training pipeline goes through. Do
not put CSV files in your `data_path` and expect binary-classification
behavior; you'll silently get a multi-label-bin path. If you have CSV data,
convert it to a pkl of one of the formats above.