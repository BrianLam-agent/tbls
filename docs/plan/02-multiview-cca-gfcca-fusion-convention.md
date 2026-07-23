# Plan 02: Multi-view CCA/GFCCA fusion convention (no real multi-view data yet)

> Status: draft, not yet handed off. **Hard dependency: Plan 01 must be
> `ACCEPTED` first** — this plan builds directly on `experiments/train.py`'s
> `_build_model`/`_cross_validate`/`_run_grid` and `experiments/hyperparams.py`
> as landed by Plan 01; do not start implementation until that lands on
> `master`.

## Goal

Establish (not "validate against real data" — there is none yet) a clean,
documented **data contract and code path** for multi-view feature fusion
(`tbls.PairwiseKCCA` / `tbls.GraphFuzzyKCCA`) inside the `experiments/`
training pipeline, so that when a genuine multi-view dataset arrives (the
user mentioned fundus-image modality fusion as the likely future case), it
only needs to be exported in the documented pkl shape — no pipeline code
changes required. Validate the wiring end-to-end with a **synthetic**
2-view dataset (clearly labeled as such), not real project data.

## Why (context)

Investigated both real datasets currently in `experiments/datasets/`
(`biomedical_larger.pkl`: `DM`/`CKD`/`BC`/`CG`, each a single flat
`(n, 204)` matrix; `data_cross_train.pkl`: 32 keys, all single flat
matrices — mostly standard `imbalanced-learn` benchmark datasets) and the
original pre-refactor `BLS/` tree the user provided: **no multi-view
structure exists anywhere, in the data or in any historical code path.**
`othercode/classifiers.py`'s `X_views`-consuming comparison classifiers
(MOFA/DIABLO/MOGONET/SNF) always assumed the caller already had `X_views`
from somewhere; nothing in this repo's history ever produced them from these
pkl files. So there is nothing to "wire up to real data" right now — the
actionable work is defining the convention and proving the plumbing, and
deferring real-data ingestion to whenever it exists (at which point it
should be a small loader addition, not a redesign).

CCA/GFCCA's role in the pipeline (for anyone reading this plan without the
chat context): given aligned per-sample feature matrices from two (or more)
modalities/views of the *same* samples, they project each view into a
shared, correlation-maximizing (CCA) or additionally label-supervised +
noise-robust (GFCCA: adds a discriminative graph-embedding term and
Intuitionistic Fuzzy Set sample-credibility weighting) space; the
concatenated projections become the fused feature matrix fed into
`TBLS`/`BroadLearningSystem` exactly like a normal `X` today.

## Design references

- [`docs/usage-cca-gfcca.md`](../usage-cca-gfcca.md) — the estimator-level
  API this plan wires into a training pipeline.
- [`docs/architecture.md`](../architecture.md) section 3 (package/experiments
  split) — the new multi-view loader is `experiments/`-only, same rationale.
- [`docs/usage-experiments-cli.md`](../usage-experiments-cli.md) — gets a new
  section; also linked from a new `docs/usage-multiview-fusion.md`.
- `docs/plan/01-tbls-graph-ifs-strategy-and-grid-search.md` — the
  `_build_model`/`_cross_validate`/`hyperparams.py` shapes this plan extends.

## Upstream dependencies

- **Plan 01** (`01-tbls-graph-ifs-strategy-and-grid-search.md`) must be
  `ACCEPTED`. This plan's `experiments/train.py` changes are additive on top
  of Plan 01's `_build_model`/`_cross_validate`/`_run_grid`/`hyperparams.py`
  — re-read those functions as actually landed (not just as drafted in Plan
  01) before starting, in case the accepted version differs from the draft.

## Deliverables for downstream work

- A documented pkl contract (below) any future multi-view dataset export
  must follow.
- `experiments/multiview.py`: the only file a follow-up plan needs to touch
  to point at real data (e.g., swap the synthetic-view generator referenced
  in tests for a real loader, once real data exists) — core wiring in
  `train.py` should not need further changes.
- `CCA_DEFAULTS`/`CCA_GRID`/`GFCCA_DEFAULTS`/`GFCCA_GRID` in
  `experiments/hyperparams.py`, seeded with the user's already-tuned point
  values.

## Current evidence and assumptions

- `experiments/dataprocess.py::DataLoader.preprocess(X_train, y_train,
  X_test)` is single-view only (one `StandardScaler`, one feature selector,
  one resampler). It is **not modified by this plan** — multi-view gets its
  own loader (`experiments/multiview.py::MultiViewDataLoader`), kept
  separate to avoid any regression risk to the already-accepted single-view
  path.
- `experiments/train.py::_load_subsets` (as landed by Plan 01) returns
  `dict[str, tuple[np.ndarray, np.ndarray]]` keyed by pkl sub-dataset key,
  requiring `sub["data"]`/`sub["target"]`. This plan adds a parallel
  multi-view-aware variant; see Step 3.
- `tbls.cca.build_cca_features`/`project_cca_features` and
  `tbls.gfcca.build_gfcca_features`/`project_cca_features` already exist and
  are unit-tested at the estimator level (`tests/test_cca.py`,
  `tests/test_gfcca.py`). **Do not re-implement fusion math** — this plan is
  pipeline wiring only.
- `tbls.gfcca.project_cca_features` and `tbls.cca.project_cca_features` are
  two *different* functions with the *same name* (see
  `docs/usage-cca-gfcca.md`) — the multi-view loader must call whichever one
  matches the models dict it holds (`cca_models` → `tbls.cca`,
  `gfcca_models` → `tbls.gfcca`), never mix them up.
- `experiments/hyperparams.py` (as landed by Plan 01) already has the
  user's `CCA_K`/`CCA_LAMBDA`/`KERNEL_GAMMA`/`GFCCA_GRAPH_GAMMA`/
  `DISCRIMINATIVE_BETA`/`GFCCA_SIGMA_IF`/`GFCCA_DELTA_IF` values as
  **commented-out** reference constants, plus a note that `GFCCA_SIGMA_GRAPH`
  is dead (the user's own words: "此参数已无用") — do not activate that one.

## Non-goals

- Ingesting any real multi-view dataset (fundus images or otherwise) — none
  exists in this repo yet. When it does, that's a follow-up plan (likely
  small: a loader adapter into the contract this plan defines).
- Image feature extraction (e.g. a CNN backbone producing per-modality
  feature vectors from raw fundus images) — out of scope; this plan assumes
  per-view feature vectors already exist, however they were produced.
- Changing `DataLoader`/single-view behavior in any way.
- SMOTE-family resampling for multi-view data (see Decision 2 below — this
  is an explicit, permanent restriction, not a deferred TODO).
- More than two views in the *test* coverage (the fusion library functions
  already support N views generically via all-pairs concatenation; this plan
  only needs to prove 2 views work, not add N-view-specific test matrices).

## Decisions (must be implemented exactly as below, not improvised)

### Decision 1: multi-view pkl contract

```python
{
    cohort_key: {
        "views": [X1, X2, ...],  # list of aligned (n_samples, n_features_i) arrays;
                                  # same n_samples across all views for this key,
                                  # row i in every view is the same underlying sample
        "target": y,             # (n_samples,)
    },
    ...
}
```

A cohort dict is multi-view if it has a `"views"` key (a list/tuple of
arrays); it is single-view (existing, untouched behavior) if it has a
`"data"` key. A single pkl file may mix single-view and multi-view cohort
keys — each key is handled independently based on which key it has. Exactly
one of `"views"`/`"data"` must be present per cohort key; a cohort dict
having neither (or both) is a hard error.

### Decision 2: resampling restrictions for multi-view data

SMOTE-family resamplers (`smote`, `adasyn`, `border_smote`, `smote_tomek`,
`smote_enn`) **synthesize new feature vectors by interpolation** — there is
no way to synthesize a "new sample" consistently across two different
feature spaces without inventing an interpolation scheme the library
doesn't expose. They are therefore **not supported for multi-view data**;
requesting one raises `ValueError` naming the unsupported resampler and
pointing at `docs/usage-multiview-fusion.md`.

Two categories are supported:

- **Index-only** (`undersample` → `RandomUnderSampler`, and a new
  `oversample` → `RandomOverSampler`, added to the multi-view loader's
  resampler map since it doesn't yet exist as a `DataLoader.RESAMPLERS`
  entry either): these select/duplicate whole existing rows based only on
  class counts, never touching feature values. Implemented by resampling a
  dummy `np.arange(n).reshape(-1, 1)` "X" against `y`, then applying the
  returned index array to every view and to `y`.
- **Reference-view** (`tomek` → `TomekLinks`): removes majority-class
  samples that form a Tomek link in *some* feature space — ambiguous across
  views, so a `preprocess.fusion_reference_view: int` config key (default
  `0`) selects which view's feature space `TomekLinks` computes links
  against; the resulting kept/removed row mask is then applied identically
  to every view and to `y`.

### Decision 3: fusion only activates when data actually has multiple views

`fusion.method` (`"cca"` | `"gfcca"` | omitted) in the YAML config only
takes effect for cohort keys loaded via the `"views"` branch. Single-view
cohort keys (all current real data) are completely unaffected by this
plan — no new config key changes their behavior. If a multi-view cohort is
loaded and `fusion.method` is omitted, default to `"gfcca"` (the tuned,
recommended default per the user's request), not `"none"` — the pipeline
should do *something* sensible without requiring the config to spell out
`fusion.method: gfcca` every time it's the desired default.

## Data model and storage changes

None beyond the pkl contract in Decision 1 (documentation only — this plan
does not create any real multi-view pkl file in the repo; see Non-goals).

## Retrieval and tool contract changes

Not applicable.

## Workflow / prompt / agent changes

Not applicable.

## Implementation steps

### Step 1 — `experiments/hyperparams.py`: activate CCA/GFCCA constants

Replace the commented block with:

```python
CCA_DEFAULTS: dict = {
    "k": 15,
    "reg_lambda": 0.1,
    "kernel_gamma": 1.0,
}
CCA_GRID: dict = {
    "k": [7, 15, 25],
    "reg_lambda": [0.01, 0.1, 1.0],
}

GFCCA_DEFAULTS: dict = {
    "k": 15,
    "reg_lambda": 0.1,
    "kernel_gamma": 1.0,
    "graph_gamma": 0.5,
    "discriminative_beta": 0.3,
    "sigma_if": 1.0,
    "delta_if": 0.5,
    # GFCCA_SIGMA_GRAPH from the original constant block is a documented dead
    # parameter (superseded by the discriminative graph; not a GraphFuzzyKCCA
    # constructor argument) -- intentionally not included here.
}
GFCCA_GRID: dict = {
    "graph_gamma": [0.1, 0.5, 1.0],
    "discriminative_beta": [0.1, 0.3, 0.5],
}
```

Same "starting example, tune here directly" docstring caveat as the
existing `BLS_GRID`/`TBLS_GRID`.

### Step 2 — `experiments/multiview.py` (new): loader + fusion wiring

```python
"""Multi-view data loading and CCA/GFCCA feature fusion for experiments/.

See docs/usage-multiview-fusion.md for the pkl contract, the resampling
restrictions, and why this is kept separate from dataprocess.py::DataLoader.
"""
```

Contents:

- `MultiViewDataLoader` class: constructor mirrors `DataLoader`'s
  `(feature_selection, resampling)` args plus `fusion_reference_view: int = 0`.
  - `preprocess_views(X_views_train, y_train, X_views_test=None) ->
    (X_views_train_processed, y_train_processed, X_views_test_processed)`:
    per-view `StandardScaler` + `feature_selection` (reuse
    `DataLoader.FEATURE_SELECTORS`, applied independently per view — import
    the dict from `dataprocess`, don't duplicate it), then resampling per
    Decision 2 (raise `ValueError` for any SMOTE-family key).
- `fuse_views(X_views_train, y_train, X_views_test, method, **fusion_kwargs)
  -> (F_train, F_test)`: `method == "cca"` → `tbls.cca.build_cca_features` +
  `tbls.cca.project_cca_features`; `method == "gfcca"` →
  `tbls.gfcca.build_gfcca_features` (needs `y_train`) +
  `tbls.gfcca.project_cca_features`. Raise `ValueError` for any other
  `method` string.
- `load_multiview_subsets(pkl_path) -> dict[str, tuple[list[np.ndarray], np.ndarray]]`:
  same shape/validity filtering as `train.py::_load_subsets` (drop label
  `-1`, binarize, `nan_to_num`) but for cohort keys with a `"views"` list;
  raise on a cohort key with neither `"views"` nor `"data"`, or with both.

### Step 3 — `experiments/train.py`: branch on single-view vs multi-view

1. Extend the per-cohort loading in `train()` to inspect the raw pkl content
   (or call both `_load_subsets` and `experiments.multiview.load_multiview_subsets`
   and merge, whichever is less invasive to the Plan-01-landed code — decide
   based on the actual landed `_load_subsets` signature, not the draft
   above) so each cohort key resolves to either the existing
   `(x, y)` single-view path (**unchanged**) or a new `(x_views, y)`
   multi-view path.
2. For a multi-view cohort key, `_cross_validate`'s per-fold body becomes:
   split every view by the same `train_idx`/`test_idx`,
   `MultiViewDataLoader.preprocess_views(...)`, then
   `fuse_views(..., method=cfg.get("fusion", {}).get("method", "gfcca"))`
   using `CCA_DEFAULTS`/`GFCCA_DEFAULTS` merged with any `fusion.*` config
   overrides (same override-merge pattern `_build_model` already uses for
   model hyperparameters), producing `F_train`/`F_test`, then continue
   exactly as today (`model.fit(F_train, y_tr)`, evaluate, save). Do not
   duplicate the whole `_cross_validate` function if the diff can be kept to
   a single branch inside it.
3. Add a `--fusion` typer option (`cca`/`gfcca`/`none` — `none` only valid
   for single-view cohorts, or a no-op if a multi-view cohort's config
   already omits fusion... actually per Decision 3, fusion is never "off" by
   choice for multi-view data since CCA/GFCCA fusion is *how* multiple views
   become one feature matrix a classifier can consume — clarify in the
   flag's help text that `--fusion` only overrides *which* fusion method,
   not whether fusion happens).
4. `--grid` for a multi-view cohort should additionally sweep
   `CCA_GRID`/`GFCCA_GRID` (whichever `fusion.method` is active) the same
   way it already sweeps `TBLS_GRID`/`BLS_GRID` — the grid becomes the
   Cartesian product of model params × fusion params. If this makes
   `_run_grid` too large a diff, an acceptable narrower scope for this plan
   is: `--grid` on a multi-view cohort sweeps *only* the model grid at a
   fixed fusion default, and sweeping fusion params is left as an explicit
   follow-up noted in the acceptance report — call this trade-off out
   explicitly if taken, don't silently ship a partial grid without saying so.

### Step 4 — Tests

`tests/test_multiview.py` (new):

- A pytest fixture generating a **synthetic** 2-view dataset: take
  `sklearn.datasets.make_classification(n_samples=120, n_features=16, ...)`
  and split its columns at index 8 into `X1 = X[:, :8]`, `X2 = X[:, 8:]` —
  **docstring/comment must say this is an arbitrary synthetic split for
  wiring validation only, not a real multi-view dataset**.
- `test_multiview_pkl_contract_detection`: a cohort dict with `"views"` is
  detected as multi-view; one with `"data"` as single-view; one with both or
  neither raises.
- `test_multiview_preprocess_views_shapes`: `MultiViewDataLoader.preprocess_views`
  returns per-view arrays with the same row count as `y`, feature selection
  applied independently per view (assert different selected-feature counts
  across two views with different informative-feature ratios).
- `test_multiview_resampling_smote_family_raises`: `resampling="smote"`
  raises `ValueError` inside `preprocess_views`.
- `test_multiview_resampling_index_based_keeps_views_aligned`:
  `resampling="oversample"`/`"undersample"` — assert all views and `y` have
  identical resampled row count and that view rows still correspond (e.g. by
  resampling a dataset where view 1's values are a deterministic function of
  view 2's, and checking the relationship still holds post-resample).
- `test_fuse_views_cca` / `test_fuse_views_gfcca`: `fuse_views` on the
  synthetic 2-view fixture produces finite `F_train`/`F_test` of the
  expected concatenated width (`2 * k` for a single view pair).
- `test_train_cli_multiview_smoke` (integration): write the synthetic
  2-view fixture to a temp pkl (`tmp_path`), point a minimal config at it,
  run `experiments.train.train(...)` (or invoke the typer `app` via
  `typer.testing.CliRunner`) with `--n-splits 2`, assert it completes and
  produces the expected Excel output — this is the multi-view analog of the
  existing `tests/test_real_dataset_smoke.py`, but synthetic since there's
  no real multi-view data to skip-if-absent.

### Step 5 — Docs

- New `docs/usage-multiview-fusion.md`: what CCA/GFCCA fusion is (reuse the
  explanation from this plan's "Why" section), the pkl contract (Decision
  1), the resampling restrictions (Decision 2) and why, the config schema,
  and a prominent note at the top: *"No real multi-view dataset exists in
  this project yet; this document describes the convention future data must
  follow, validated here only against a synthetic 2-view fixture."*
- `docs/usage-experiments-cli.md`: short new section linking to the above,
  covering `--fusion` and the multi-view `--grid` behavior.
- `README.md`: add the new doc to the documentation index table.
- `experiments/datasets/README.md`: add the multi-view pkl contract next to
  the existing single-view one, so whoever prepares the real dataset (e.g.
  the fundus-image export) has it in the same place as the rest of the
  dataset-loading documentation.

## Verification commands and test cases

```bash
uv run pytest tests/ -v          # includes new tests/test_multiview.py
uv run ruff check .
uv run ruff format --check .
uv run mypy src/tbls              # unchanged scope: experiments/ not covered by strict mypy
uv run --group experiments python -c "
from experiments.multiview import load_multiview_subsets, MultiViewDataLoader, fuse_views
print('multiview module imports OK')
"
```

No real-data manual verification step is possible or expected for this plan
(there is no real multi-view dataset) — the synthetic integration test in
Step 4 is the acceptance bar. Do not claim "verified against real data" in
the acceptance report; explicitly state it is synthetic-only, pending real
data.

## Acceptance checklist

- [ ] Plan 01 is `ACCEPTED` before this plan starts implementation.
- [ ] Multi-view pkl contract (Decision 1) implemented exactly as specified;
      a cohort with both/neither of `"views"`/`"data"` raises a clear error.
- [ ] SMOTE-family resamplers raise `ValueError` for multi-view data;
      `undersample`/`oversample`/`tomek` work and keep views aligned
      (verified by the deterministic-relationship test in Step 4).
- [ ] `fuse_views` correctly dispatches `cca_models` to `tbls.cca.project_cca_features`
      and `gfcca_models` to `tbls.gfcca.project_cca_features` — never mixed up.
- [ ] Single-view cohorts (all current real datasets) are provably
      unaffected: existing single-view tests
      (`tests/test_real_dataset_smoke.py`, existing `experiments/train.py`
      behavior) still pass unmodified.
- [ ] `experiments/hyperparams.py` has `CCA_DEFAULTS`/`CCA_GRID`/
      `GFCCA_DEFAULTS`/`GFCCA_GRID`; `GFCCA_SIGMA_GRAPH` is not present as a
      live key anywhere.
- [ ] New `docs/usage-multiview-fusion.md` exists, linked from `README.md`
      and `docs/usage-experiments-cli.md`, and states plainly that it is
      validated only against synthetic data so far.
- [ ] Acceptance report explicitly states this was validated with a
      synthetic 2-view fixture only, not real data, and names the exact
      follow-up needed once real (e.g. fundus-image) multi-view data exists.

## Suggested commits

1. `feat(experiments): activate CCA/GFCCA hyperparameter defaults and grids`
2. `feat(experiments): add MultiViewDataLoader and CCA/GFCCA fusion wiring`
3. `feat(experiments): multi-view branch in train.py CLI (--fusion, --grid)`
4. `test(experiments): synthetic multi-view fixture and end-to-end smoke test`
5. `docs: multi-view fusion convention and usage`
