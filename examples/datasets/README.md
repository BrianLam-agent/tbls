# Example datasets (git-ignored)

Real dataset `.pkl` files for the `examples/` ablation runs live here. They are
**git-ignored** (large binary files — see the root `.gitignore`
`examples/datasets/*.pkl` rule) and must be present on the local machine to run
the example configs under `examples/configs/`.

## Setup

The example configs (`examples/configs/tbls_*.yaml`) set
`data_path: examples/datasets/`, so `train.py` reads
`examples/datasets/biomedical_larger.pkl`. Copy the canonical `.pkl` from
`experiments/datasets/` here (a symlink would avoid the second 26 MB copy, but
creating one on Windows needs an elevated shell or Developer Mode enabled, so a
plain copy is the zero-friction option):

```bash
# from the repo root (any shell):
cp experiments/datasets/biomedical_larger.pkl examples/datasets/
```

## Expected pkl format

Same contract as `experiments/datasets/` (see that directory's `README.md`):
`joblib.load(...)` returns either a flat `{"data": X, "target": y}` dict or a
multi-key dict of such sub-datasets; samples with label `-1` are dropped; labels
are binarized to `{0, 1}`.

The example configs use `biomedical_larger.pkl`'s `"DM"` and `"CKD"` cohorts
(the pkl is multi-key). Point `data_path` at any other pkl that follows the
same contract to reuse the configs for a different dataset.