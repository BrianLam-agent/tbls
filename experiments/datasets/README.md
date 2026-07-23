# Experiment datasets

Real dataset files live here. They are **git-ignored** (large binary files) and
must be present on the local machine to run the real-dataset smoke test and
`experiments/train.py`.

## Files

- `biomedical_larger.pkl` (~26 MB) — moved here from the legacy `dataset/`
  directory during the package refactor.
- `data_cross_train.pkl` (~335 MB) — moved here from the legacy `dataset/`
  directory.

## Expected pkl format

`joblib.load(...)` returns a dict. `experiments/dataprocess.py::DataLoader`
accepts either:

- a top-level dict with `data` and `target` keys, or
- a dict of sub-datasets where each value is a dict with `data` and `target`
  keys (multi-key datasets are iterated by `experiments/train.py`).

Samples with label `-1` are filtered out as invalid.

## Reproducing

Place the two `.pkl` files in this directory, then run:

```bash
uv run --group experiments python experiments/smoke_run.py
```
