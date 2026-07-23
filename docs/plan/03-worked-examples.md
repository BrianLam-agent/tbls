# Plan 03: Worked examples (`examples/`)

> Status: final, ready to hand off. No hard predecessor (both prior plans are
> `ACCEPTED`).

## Goal

Add a small, runnable `examples/` directory (outside the published package,
like `experiments/`) with two scripts that demonstrate the currently-accepted
capabilities end to end, using the real `experiments/datasets/*.pkl` data:

1. A single TBLS training run.
2. The same, with `--grid` hyperparameter search.

These are documentation-by-example, not tests (tests already cover
correctness; these exist so a new user/reviewer can see the intended usage
pattern in under a minute, matching the "how do I actually use this" ask).

## Design references

- `docs/usage-tbls.md`, `docs/usage-experiments-cli.md` — the APIs being
  demonstrated; do not duplicate their content, link to them from `examples/README.md`.
- `experiments/smoke_run.py` — existing precedent for a minimal runnable
  script against real data; reuse its dataset-loading pattern rather than
  reinventing one.

## Non-goals

- Packaging `examples/` for PyPI (not part of the wheel, same as `experiments/`).
- A multi-view fusion example (no real multi-view dataset exists yet; revisit
  once one does).
- Any new library or CLI functionality — this plan only adds example scripts
  that call existing, accepted APIs.

## Implementation steps

### Step 1 — `examples/01_train_tbls.py`

A short, heavily-commented script (not a typer CLI — plain `if __name__ ==
"__main__"`) that:

- Loads `experiments/datasets/biomedical_larger.pkl`, cohort key `"DM"`
  (reuse `experiments/dataprocess.py::DataLoader` for scaling/feature
  selection, a plain `sklearn.model_selection.train_test_split`, no CV — keep
  this example minimal).
- Fits `tbls.TBLS()` with its defaults (i.e. `graph_strategy="discriminative"`,
  `if_strategy="simple"`, `use_if_weights=True`, `graph_gamma=0.1`).
- Prints accuracy, balanced accuracy, and macro F1 on the held-out split via
  `experiments.evaluate.TBLSEvaluator.calculate_metrics`.
- A top-of-file docstring explaining prerequisites (`uv sync --group experiments`,
  the dataset must already be in `experiments/datasets/`) and expected runtime
  (a few seconds).

### Step 2 — `examples/02_train_tbls_with_grid_search.py`

Same dataset/cohort, but drives `experiments/train.py`'s CLI programmatically
(import `experiments.train` and call its internals the same way
`tests/test_experiments_train.py` already does — do not shell out to a
subprocess) with `--grid`-equivalent behavior: build a small 2x2 grid (reuse
`experiments/hyperparams.py::TBLS_GRID` or a smaller inline example grid to
keep runtime short), run it, and print the ranked `GridSummary` rows to
stdout at the end (no Excel writing required for the example itself, though
it may reuse `TBLSResultSaver` if that's less code than reimplementing
output — either is acceptable).

### Step 3 — `examples/README.md`

- Prerequisites (`uv sync --group experiments`, real dataset present).
- One `bash` command per script.
- Expected (approximate, not pinned-exact) output for each, so a reader can
  sanity-check their own run without needing to actually execute it.
- Links to `docs/usage-tbls.md` and `docs/usage-experiments-cli.md` for the
  full API/CLI reference (do not repeat that content here).

### Step 4 — Root `README.md`

Add a one-line pointer to `examples/` near the top (e.g. in the "Quick start"
or install section) so it's discoverable without reading the full doc index.

## Verification commands

```bash
uv run --group experiments python examples/01_train_tbls.py
uv run --group experiments python examples/02_train_tbls_with_grid_search.py
uv run ruff check examples/
uv run ruff format --check examples/
```

Both scripts must run to completion against the real dataset and print sane
metrics (accuracy/balanced accuracy roughly consistent with the smoke-test
reference in `docs/usage-experiments-cli.md`, i.e. not near-random and not
suspiciously perfect on held-out data).

## Acceptance checklist

- [ ] Both scripts run end-to-end against real data, no crashes, no warnings
      beyond the known joblib/numpy deprecation noise already present elsewhere.
- [ ] Neither script duplicates `experiments/train.py`'s CV/grid logic by
      copy-paste — they import and call it, or use `DataLoader`/`TBLSEvaluator`
      directly for the minimal single-run case.
- [ ] `examples/README.md` present with accurate example output.
- [ ] `ruff check`/`ruff format --check` pass on `examples/`.

## Suggested commits

1. `docs(examples): single TBLS training run against real data`
2. `docs(examples): TBLS grid-search example`
3. `docs(examples): README + root README pointer`
