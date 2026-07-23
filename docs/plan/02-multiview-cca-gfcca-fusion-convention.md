# Plan 02: Multi-view CCA/GFCCA fusion pipeline wiring

> Status: **final, ready to hand off**. **Hard dependency: Plan 01 must be
> `ACCEPTED` first** (it is, as of commit `8effeaf`) — this plan builds on
> `experiments/train.py`'s `_build_model`/`_cross_validate`/`_run_grid` and
> `experiments/hyperparams.py` as landed by Plan 01.
>
> **The data contract and configuration schema are already finalized and
> documented in [`docs/usage-multiview-fusion.md`](../usage-multiview-fusion.md)
> and [`experiments/datasets/README.md`](../../experiments/datasets/README.md).
> Implement exactly what those documents specify — do not redesign the
> contract while implementing this plan.** If something in those documents
> is ambiguous or turns out to be impractical while implementing, stop and
> ask; do not silently deviate.

## Goal

Implement the multi-view loading, per-view preprocessing, fusion-group
dispatch, and CLI wiring described in `docs/usage-multiview-fusion.md`,
validated with a synthetic 2-view fixture (there is no real multi-view
dataset yet — see that document's Section 6).

## Why (context)

See `docs/usage-multiview-fusion.md` Section 1 for what CCA/GFCCA fusion is
and why it's needed; this plan is the *pipeline wiring*, not new fusion math
— `tbls.cca.build_cca_features`/`tbls.gfcca.build_gfcca_features` already
exist, are unit-tested, and must not be modified by this plan.

## Design references

- **[`docs/usage-multiview-fusion.md`](../usage-multiview-fusion.md) — the
  authoritative spec for everything in this plan**: pkl contract (dict of
  named views, not a list), preprocessing order, resampling restrictions,
  fusion groups, config schema. Read it in full before writing any code.
- [`docs/usage-cca-gfcca.md`](../usage-cca-gfcca.md) — the estimator-level
  API (`PairwiseKCCA`, `GraphFuzzyKCCA`, `build_cca_features`,
  `build_gfcca_features`, `project_cca_features`) this plan wires into a
  training pipeline; do not modify these functions.
- [`docs/architecture.md`](../architecture.md) section 3 (package/experiments
  split) — the new multi-view loader is `experiments/`-only, same rationale
  as `experiments/dataprocess.py`.
- `docs/plan/01-tbls-graph-ifs-strategy-and-grid-search.md` — the
  `_build_model`/`_cross_validate`/`_run_grid`/`hyperparams.py` shapes this
  plan extends; re-read the *landed* code (commit `8effeaf`), not the plan
  draft, since implementation details may have settled slightly differently.

## Non-goals

- Ingesting any real multi-view dataset — none exists. When one is exported
  in the `docs/usage-multiview-fusion.md` format, that's a follow-up task
  (should be small: point a config at the new `.pkl`), not a new plan.
- Reading raw images or any other raw modality into feature vectors —
  explicitly out of scope everywhere in this project (see that document's
  header note).
- Changing `experiments/dataprocess.py::DataLoader` or its single-view
  behavior in any way. Single-view cohorts must be provably unaffected.
- Changing `tbls.cca`/`tbls.gfcca` fusion math, `PairwiseKCCA`, or
  `GraphFuzzyKCCA` in any way.
- Sweeping fusion hyperparameters via `--grid` in the same pass as model
  hyperparameters — see Step 4's narrower scope note; call out explicitly if
  taken, rather than silently shipping a partial grid.

## Upstream dependencies

Plan 01, `ACCEPTED` (commit `8effeaf` on `master`).

## Deliverables

- `experiments/multiview.py` (new): the only file a future "point at real
  data" follow-up should need to touch.
- `experiments/hyperparams.py`: `CCA_DEFAULTS`/`CCA_GRID`/
  `GFCCA_DEFAULTS`/`GFCCA_GRID`, replacing the currently-commented block.
- `experiments/train.py`: multi-view branch (auto-detected per cohort key
  from the pkl shape — see contract), `--fusion` CLI override.
- `tests/test_multiview.py` (new).
- Already done (do not redo): `docs/usage-multiview-fusion.md`,
  `experiments/datasets/README.md`'s multi-view section, `README.md`'s doc
  index entry.

## Implementation steps

### Step 1 — `experiments/hyperparams.py`: activate CCA/GFCCA constants

Replace the commented block with:

```python
CCA_DEFAULTS: dict = {
    "cca_k": 15,
    "cca_lambda": 0.1,
    "kernel_gamma": 1.0,
}
CCA_GRID: dict = {
    "cca_k": [7, 15, 25],
    "cca_lambda": [0.01, 0.1, 1.0],
}

GFCCA_DEFAULTS: dict = {
    "cca_k": 15,
    "cca_lambda": 0.1,
    "kernel_gamma": 1.0,
    "graph_gamma": 0.5,
    "discriminative_beta": 0.3,
    "sigma_if": 1.0,
    "delta_if": 0.5,
    # sigma_graph is a documented-dead GraphFuzzyKCCA parameter (reserved,
    # unused) -- intentionally not included here.
}
GFCCA_GRID: dict = {
    "graph_gamma": [0.1, 0.5, 1.0],
    "discriminative_beta": [0.1, 0.3, 0.5],
}
```

Keyword names must match `build_cca_features`/`build_gfcca_features`'s actual
parameter names exactly (`cca_k`, `cca_lambda`, `kernel_gamma`, `graph_gamma`,
`discriminative_beta`, `sigma_if`, `delta_if` — verify against the current
`src/tbls/cca.py`/`src/tbls/gfcca.py` signatures, not this plan text, in case
they've since changed) since Step 2 passes these dicts as `**kwargs` directly.
Same "starting example, tune here directly" docstring caveat as the existing
`BLS_GRID`/`TBLS_GRID`.

### Step 2 — `experiments/multiview.py` (new)

```python
"""Multi-view data loading and CCA/GFCCA feature fusion for experiments/.

See docs/usage-multiview-fusion.md for the pkl contract, fusion-group
config, and resampling restrictions this module implements. Do not change
the contract here without updating that document first.
"""
```

- `load_multiview_cohort(pkl_path, cohort_key) -> tuple[dict[str, np.ndarray], np.ndarray]`:
  loads one cohort's `{"views": {name: X, ...}, "target": y}`, applying the
  same label filtering (`target != -1`, binarize, `nan_to_num`) as
  `experiments/train.py::_load_subsets`. Raise a clear `ValueError` if a
  cohort dict has both/neither of `"data"`/`"views"`.
- `MultiViewDataLoader(feature_selection=None, resampling=None, fusion_reference_view=None)`:
  - `preprocess_views(X_views_train: dict[str, np.ndarray], y_train, X_views_test: dict[str, np.ndarray] | None) -> tuple[dict, np.ndarray, dict | None]`:
    per-view `StandardScaler` + `feature_selection` (import
    `DataLoader.FEATURE_SELECTORS` from `dataprocess`, do not duplicate it),
    each view getting its own fitted scaler/selector instance, keyed by view
    name so train/test use matching instances.
  - Resampling per `docs/usage-multiview-fusion.md` Section 3:
    - SMOTE-family (`smote`, `adasyn`, `border_smote`, `smote_tomek`,
      `smote_enn`) → `ValueError` naming the resampler and pointing at
      `docs/usage-multiview-fusion.md`.
    - `oversample`/`undersample` → resample a dummy
      `np.arange(n).reshape(-1, 1)` "X" against `y`, apply the returned row
      indices to every view (by name) and to `y`. Add `RandomOverSampler` as
      `"oversample"` here (it is not currently a `DataLoader.RESAMPLERS` key
      either — do not add it to `DataLoader` itself, only here).
    - `tomek` → run `TomekLinks` against the view named by
      `fusion_reference_view` (required, config error if unset when
      `resampling == "tomek"` and there is more than one view), apply the
      resulting keep-mask to every view and to `y`.
- `fuse_views(X_views: dict[str, np.ndarray], y: np.ndarray | None, X_views_test: dict[str, np.ndarray] | None, method: str, view_groups: list[list[str]] | None, **fusion_kwargs) -> tuple[np.ndarray, np.ndarray | None]`:
  - Validate `view_groups` is a partition of `X_views.keys()` (every key in
    exactly one group); default to `[[*sorted(X_views.keys())]]` if `None`.
  - For each group: if `len(group) == 1`, passthrough (that view's own
    columns, train and test, unchanged — no CCA/GFCCA call). If
    `len(group) >= 2`, call `build_cca_features`/`build_gfcca_features` (per
    `method`) on `[X_views[name] for name in group]`, then
    `project_cca_features` (the one matching `method` — `tbls.cca`'s for
    `"cca"`, `tbls.gfcca`'s for `"gfcca"`; **never mix the two** — this
    footgun is documented in Plan 01's precedent and must be avoided the
    same way here) on the test-side views for the same group.
  - Concatenate all groups' train blocks into `F_train`, all groups' test
    blocks (if `X_views_test` given) into `F_test`.
  - Raise `ValueError` for an unknown `method`.

### Step 3 — `experiments/train.py`: branch on single-view vs multi-view

1. Extend cohort loading so each cohort key resolves to either the existing
   `(x, y)` single-view path (**unchanged**) or a `(x_views: dict, y)`
   multi-view path, based on `"data"` vs `"views"` in the raw pkl content —
   reuse `experiments.multiview.load_multiview_cohort` rather than
   duplicating its validation.
2. For a multi-view cohort key, the per-fold body: split every view (by
   name) and `y` by the same `train_idx`/`test_idx`,
   `MultiViewDataLoader.preprocess_views(...)`, then
   `fuse_views(..., method=cfg["fusion"].get("method", "gfcca"), view_groups=cfg["fusion"].get("view_groups"))`
   using `CCA_DEFAULTS`/`GFCCA_DEFAULTS` merged with any `fusion.*` config
   overrides (same override-merge pattern `_build_model` already uses),
   producing `F_train`/`F_test`, then continue exactly as today
   (`model.fit(F_train, y_tr)`, evaluate, save). Keep the diff to a branch
   inside the existing per-fold function rather than a parallel copy of it,
   if the landed Plan 01 code structure allows.
3. Add a `--fusion [cca|gfcca]` typer option overriding `fusion.method` for
   multi-view cohorts; document in its help text that it only overrides
   *which* fusion method runs, not whether fusion happens (fusion always
   runs for a multi-view cohort — it's how multiple views become one
   feature matrix a classifier can consume).
4. `--grid` scope for this plan: sweep **only** the model grid
   (`TBLS_GRID`/`BLS_GRID`) at a fixed fusion default for multi-view
   cohorts; do not also sweep `CCA_GRID`/`GFCCA_GRID` in this pass. State
   this explicitly in the acceptance report as a known, intentional scope
   limit (sweeping fusion hyperparameters too is a reasonable follow-up, not
   silently dropped — just not in this plan, to keep the diff reviewable).

### Step 4 — Tests: `tests/test_multiview.py` (new)

- A pytest fixture generating a **synthetic** 2-view dataset: split
  `sklearn.datasets.make_classification(n_samples=120, n_features=16, ...)`'s
  columns at index 8 into `{"view_a": X[:, :8], "view_b": X[:, 8:]}` —
  docstring/comment must state this is an arbitrary synthetic split for
  wiring validation only, not a real multi-view dataset.
- `test_load_multiview_cohort_contract`: a cohort dict with `"views"` loads
  correctly; one with `"data"` is rejected by this loader (it's the
  single-view path's job); one with both or neither raises `ValueError`.
- `test_preprocess_views_independent_per_view`: `MultiViewDataLoader.preprocess_views`
  returns per-view arrays with the same row count as `y`; feature selection
  applied independently (assert different selected-feature counts across two
  views constructed with different informative-feature ratios).
- `test_resampling_smote_family_raises`: `resampling="smote"` raises
  `ValueError` inside `preprocess_views` for multi-view data.
- `test_resampling_index_based_keeps_views_aligned`: `"oversample"`/
  `"undersample"` — all views and `y` end with identical row count; verify
  alignment held by resampling a dataset where `view_b`'s values are a
  deterministic function of `view_a`'s, and checking the relationship still
  holds post-resample.
- `test_fuse_views_single_group_cca` / `test_fuse_views_single_group_gfcca`:
  default `view_groups=None` on the 2-view fixture produces finite
  `F_train`/`F_test` of the expected width.
- `test_fuse_views_groups_partition_validation`: a `view_groups` covering a
  view twice, or missing one, raises `ValueError`.
- `test_fuse_views_passthrough_singleton_group`: a 3-view fixture
  (`view_a`, `view_b`, `view_c`) with `view_groups=[["view_a","view_b"],["view_c"]]`
  — assert the singleton group's output columns equal `view_c`'s
  preprocessed columns exactly (no CCA/GFCCA call happened for it).
- `test_train_cli_multiview_smoke` (integration): write the synthetic
  2-view fixture to a temp pkl (`tmp_path`), point a minimal config at it
  (with a `fusion` block), run the CLI (`typer.testing.CliRunner` or direct
  function call, matching how `tests/test_experiments_train.py` already
  invokes `train.py` internals) with `--n-splits 2`, assert it completes and
  produces the expected Excel output.

### Step 5 — Docs

Already done (`docs/usage-multiview-fusion.md`, `experiments/datasets/README.md`,
`README.md`'s doc index). Update `docs/usage-experiments-cli.md` with a short
section covering `--fusion` and the multi-view `--grid` scope limit from
Step 3.4, linking to `docs/usage-multiview-fusion.md` rather than repeating
its contract. **Do not touch any `.zh-CN.md` file** — Chinese translations
are handled separately.

## Verification commands and test cases

```bash
uv run pytest tests/ -v          # includes new tests/test_multiview.py
uv run ruff check .
uv run ruff format --check .
uv run mypy src/tbls              # unchanged scope: experiments/ not covered by strict mypy
uv run --group experiments python -c "
from experiments.multiview import load_multiview_cohort, MultiViewDataLoader, fuse_views
print('multiview module imports OK')
"
```

No real-data manual verification is possible (no real multi-view dataset
exists) — the synthetic integration test in Step 4 is the acceptance bar.
State plainly in the acceptance report that this is synthetic-only,
consistent with `docs/usage-multiview-fusion.md` Section 6.

## Acceptance checklist

- [ ] Implementation matches `docs/usage-multiview-fusion.md` exactly (pkl
      contract, resampling restrictions, fusion-group partition rules,
      passthrough-singleton behavior, config schema) — any deviation is
      flagged, not silently made.
- [ ] SMOTE-family resamplers raise `ValueError` for multi-view data;
      `oversample`/`undersample`/`tomek` keep views aligned (verified by the
      deterministic-relationship test).
- [ ] `fuse_views` dispatches `cca` results through `tbls.cca.project_cca_features`
      and `gfcca` results through `tbls.gfcca.project_cca_features` — never
      mixed up (this exact footgun was called out in this plan).
- [ ] Fusion-group partition validation rejects a view used twice or missing
      from every group; a singleton group is proven to be pure passthrough
      (exact column equality test, not just "no crash").
- [ ] Single-view cohorts (all current real datasets) are provably
      unaffected: `tests/test_real_dataset_smoke.py` and existing
      `experiments/train.py` behavior pass unmodified.
- [ ] `experiments/hyperparams.py` has `CCA_DEFAULTS`/`CCA_GRID`/
      `GFCCA_DEFAULTS`/`GFCCA_GRID` with keyword names verified against the
      actual `build_cca_features`/`build_gfcca_features` signatures.
- [ ] `docs/usage-experiments-cli.md` (English only) updated for `--fusion`
      and the `--grid` scope limit; no `.zh-CN.md` file touched.
- [ ] Acceptance report explicitly states synthetic-only validation and
      names the exact follow-up needed once real multi-view data exists.

## Suggested commits

1. `feat(experiments): activate CCA/GFCCA hyperparameter defaults and grids`
2. `feat(experiments): add MultiViewDataLoader, fusion groups, and CCA/GFCCA wiring`
3. `feat(experiments): multi-view branch in train.py CLI (--fusion)`
4. `test(experiments): synthetic multi-view fixture and end-to-end smoke test`
5. `docs: multi-view fusion CLI usage (English only)`
