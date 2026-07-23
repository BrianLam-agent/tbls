# Acceptance report: Plan 02 - Multi-view CCA/GFCCA fusion pipeline wiring

- **Plan:** `docs/plan/02-multiview-cca-gfcca-fusion-convention.md`
- **Execution date:** 2026-07-24
- **Branch:** `master`
- **Baseline commit:** `8effeaf` (Plan 01 - graph/IFS strategy switch + grid search).
- **Implementation commits:** see "Implementation commits" below (5 commits, on `master`).
- **Conclusion:** **IMPLEMENTED** - all implementation work and required
  verification complete and committed; pending reviewer acceptance. Not
  `ACCEPTED` (acceptance requires the reviewer process defined by `AGENTS.md`).

## Execution-gate note (flagged, not blocking)

Plan 02 declares Plan 01 as a hard predecessor that must be `ACCEPTED`. Plan 01
is committed on `master` (`8effeaf`, reachable from HEAD) but has **no acceptance
report** under `docs/plan/reports/` and no `execution-graph.md` entry - by the
strict plan-exec rule ("treat a missing report as not accepted") it is not
formally `ACCEPTED`. The user explicitly invoked `/plan-exec` on Plan 02, and
Plan 02's status line states Plan 01 "is ACCEPTED (commit `8effeaf`)"; per
governance ("a newer explicit user instruction may narrow or override the
requested operation") the user's invocation overrides the missing-report gate.
This gap (Plan 01 has no report/graph entry) is flagged for the reviewer to
close; it does not affect Plan 02's correctness.

## Summary

The multi-view CCA/GFCCA fusion pipeline is wired into `experiments/` per
`docs/usage-multiview-fusion.md` (the finalized contract, already in the working
tree). A multi-view cohort (`{"views": {...}, "target": y}`) is auto-detected
per cohort key from the pkl shape; per-view preprocessing (StandardScaler +
feature selection) and row-aligned resampling run via `MultiViewDataLoader`,
then `fuse_views` dispatches CCA/GFCCA fusion groups (singleton group =
passthrough), and the fused matrix is fed to `TBLS`/`BroadLearningSystem`
exactly like single-view `X`. Single-view cohorts are provably unaffected.
Validated end-to-end with a **synthetic** 2-view fixture - no real multi-view
dataset exists (consistent with `docs/usage-multiview-fusion.md` Section 6).

## Implementation commits

| Hash | Subject |
|------|---------|
| (1) | `feat(experiments): activate CCA/GFCCA hyperparameter defaults and grids` |
| (2) | `feat(experiments): add MultiViewDataLoader, fusion groups, and CCA/GFCCA wiring` |
| (3) | `feat(experiments): multi-view branch in train.py CLI (--fusion)` |
| (4) | `test(experiments): synthetic multi-view fixture and end-to-end smoke test` |
| (5) | `docs: multi-view fusion CLI usage (English only)` |

(Exact hashes recorded in `git log`; the report and execution-graph commits
follow.)

## Files / interfaces changed

- **`experiments/hyperparams.py`** (Step 1): replaced the commented
  `CCA_*`/`GFCCA_*` block with active `CCA_DEFAULTS`/`CCA_GRID`/
  `GFCCA_DEFAULTS`/`GFCCA_GRID`. Keyword names verified against the actual
  `tbls.cca.build_cca_features` (`cca_k`, `cca_lambda`, `kernel_gamma`) and
  `tbls.gfcca.build_gfcca_features` (`cca_k`, `cca_lambda`, `kernel_gamma`,
  `graph_gamma`, `discriminative_beta`, `sigma_if`, `delta_if`, ...) signatures;
  `sigma_graph` (documented-dead) intentionally omitted.
- **`experiments/multiview.py`** (new, Step 2):
  - `load_multiview_cohort(pkl_path, cohort_key)` - loads one cohort's
    `{"views":..., "target":y}`, applies the same label filtering/binarization/
    `nan_to_num` as `train.py::_load_subsets`; raises `ValueError` for both/
    neither of `"data"`/`"views"`.
  - `MultiViewDataLoader.preprocess_views(...)` - per-view `StandardScaler` +
    `feature_selection` (reuses `DataLoader.FEATURE_SELECTORS`, not duplicated),
    each view its own fitted scaler/selector keyed by name; resampling per
    Section 3 (SMOTE-family -> `ValueError`; `oversample`/`undersample` ->
    index-only via `sample_indices_`, applied identically to every view + y;
    `tomek` -> reference-view `TomekLinks`, keep-mask applied to every view).
    `RandomOverSampler` added as `"oversample"` here only (not in `DataLoader`).
  - `fuse_views(...)` - validates `view_groups` is a partition (duplicated/
    missing/extra view -> `ValueError`), default `[[*sorted(keys)]]`; singleton
    group = pure passthrough (no CCA/GFCCA); `method="cca"` dispatches through
    `tbls.cca.build_cca_features` + `tbls.cca.project_cca_features`,
    `method="gfcca"` through `tbls.gfcca.*` - **never mixed** (the documented
    footgun); unknown method -> `ValueError`.
- **`experiments/train.py`** (Step 3): added `_load_cohorts` (tags each cohort
  single-view `(X,y)` vs multi-view `(views,y,"multiview")` by pkl content),
  `_fusion_kwargs` (merges `CCA_DEFAULTS`/`GFCCA_DEFAULTS` with `fusion.*`
  overrides, filtered to valid kwargs), generalized `_cross_validate` to branch
  on cohort type (multi-view: split all views + y by the same fold indices,
  `preprocess_views`, `fuse_views`, then the unchanged model-fit block), and a
  `--fusion [cca|gfcca]` typer option overriding `fusion.method`. `_run_grid`
  signature changed to take a `cohort` (the existing Plan-01 grid test was
  updated to match). `--grid` sweeps only the model grid at a fixed fusion
  default for multi-view cohorts (scope limit, documented).
- **`tests/test_multiview.py`** (new, Step 4): 10 tests - load contract
  (views/data/both/neither), per-view independent feature selection, SMOTE-family
  raises, index-only resampling keeps views aligned (verified by recovering
  original rows via each view's scaler and checking the `view_b = view_a + 1`
  relationship survives), single-group CCA + GFCCA, partition validation
  (duplicated/missing view), singleton passthrough (exact column equality),
  end-to-end CLI smoke on a temp multi-view pkl.
- **`tests/test_experiments_train.py`**: one-line update (`_run_grid` now takes
  a cohort tuple) - the Plan-01 grid smoke test.
- **`docs/usage-experiments-cli.md`** (Step 5, English only): new "Multi-view
  fusion and `--fusion`" section + the multi-view `--grid` scope limit; fixed
  the now-stale "CCA_*/GFCCA* kept commented" sentence.
- **Already in the working tree (not redone):** `docs/usage-multiview-fusion.md`
  (the contract), `experiments/datasets/README.md` multi-view section, `README.md`
  doc-index entry, the Plan 02 revision itself.
- **`src/tbls/`**: untouched. `tbls.cca`/`tbls.gfcca` fusion math was not
  modified (Non-goal).

## Plan-step evidence

- **Step 1**: `CCA_DEFAULTS`/`CCA_GRID`/`GFCCA_DEFAULTS`/`GFCCA_GRID` active in
  `hyperparams.py`; kwargs verified against signatures (see above).
- **Step 2**: `experiments/multiview.py` implements the contract; the 10
  `tests/test_multiview.py` cases pass.
- **Step 3**: `--fusion` appears in `train.py --help`; multi-view CLI run on a
  synthetic pkl completes (see "Manual verification" below); single-view real
  data still runs unchanged.
- **Step 4**: `pytest tests/test_multiview.py` -> 10 passed.
- **Step 5**: `docs/usage-experiments-cli.md` updated; no `.zh-CN.md` touched
  (per the plan).

## Verification commands and observed output

Environment: Windows 11, `uv 0.9.21`, CPython 3.13, numpy 2.5.1, scipy 1.18.0,
scikit-learn 1.9.0, imbalanced-learn 0.14.2.

| Command | Result |
|---|---|
| `uv run pytest tests/ -v` | `55 passed, 22 warnings in 5.82s` (was 45 pre-Plan-02; +10 multiview) |
| `uv run ruff check .` | `All checks passed!` |
| `uv run ruff format --check .` | `36 files already formatted` |
| `uv run mypy src/tbls` | `Success: no issues found in 19 source files` (scope unchanged: experiments/ not under strict mypy) |
| `uv build && uvx twine check dist/*` | wheel + sdist built; both `PASSED` |
| `uv run --group experiments python experiments/smoke_run.py` (single-view regression) | `TBLS smoke check OK \| key=DM acc=0.9208 macro_f1=0.8985 bls_acc=0.8387 ...` (unchanged) |
| `uv run --group experiments python experiments/train.py --n-splits 2` (single-view real data) | 4 cohorts run with sane metrics (unchanged behavior) |

### Manual verification (synthetic multi-view, the acceptance bar)

No real multi-view dataset exists. A synthetic 2-view pkl (arbitrary column
split of `make_classification`, clearly labeled synthetic) was written to
`experiments/datasets/synth_mv.pkl` and run through the CLI:

```
$ uv run --group experiments python experiments/train.py --config <mv.yaml>
INFO dataset=synth_mv model=tbls keys=['cohort'] grid=False
INFO === synth_mv / cohort : multiview views={'view_a': (120, 8), 'view_b': (120, 8)} y=(120,) ===
INFO dataset=synth_mv key=cohort fold=1/2 acc=0.6333
INFO dataset=synth_mv key=cohort fold=2/2 acc=0.6500
INFO dataset=synth_mv key=cohort avg={'avg_accuracy': 0.6417, ...}
```

`--grid` on the same multi-view pkl sweeps the model grid at the fixed CCA
fusion default (27 points), confirming the Step-3.4 scope limit:
```
INFO dataset=synth_mv key=cohort grid 1/27 {'n_enhance_trees':10,'n_map_trees':10,'reg_param':1e-08} acc=0.6583
...
```

The SMOTE-family restriction fires correctly when a multi-view cohort is run
with the default config (`preprocess.resampling: smote`):
```
ValueError: resampling='smote' is a SMOTE-family resampler ... unsupported for
multi-view data. Use 'oversample'/'undersample' (index-only) or 'tomek'
(reference-view). See docs/usage-multiview-fusion.md Section 3.
```

The synthetic pkl and `results_dir/` were removed after verification (both
git-ignored).

## Deviations from the plan (and why)

1. **`_run_grid` signature changed** to take a `cohort` (tagged single-view or
   multi-view) instead of separate `(x, y)`, so the grid path can run on a
   multi-view cohort. The one Plan-01 test that called `_run_grid(cfg, x, y, ...)`
   directly was updated to `_run_grid(cfg, (x, y), ...)` - a minimal, in-scope
   adaptation to keep the suite green; no behavior change for single-view.
2. **Resampling index recovery uses `sampler.sample_indices_`** (set by
   `RandomOverSampler`/`RandomUnderSampler`/`TomekLinks` after `fit_resample`)
   rather than a custom index-matching helper - simpler and exact. An earlier
   draft had a `_match_resample_indices` fallback; it was removed as dead code
   once `sample_indices_` was confirmed available on all three resamplers in this
   imbalanced-learn version.
3. **The plan's exact `python -c "from experiments.multiview import ..."`
   verification command** fails when run from the repo root, because
   `experiments/` modules use script-style sibling imports
   (`from dataprocess import ...`, matching the existing `experiments/train.py`
   pattern) and `python -c` does not put `experiments/` on `sys.path`. The module
   imports correctly everywhere it is actually used: via
   `uv run python experiments/train.py ...` (Python adds the script's dir to
   `sys.path[0]`) and via pytest (`tests/conftest.py` adds `experiments/` to
   `sys.path`). This is a pre-existing pattern, not a Plan-02 defect; documented
   here for completeness.

## Remaining risks / external actions / user decisions

- **Synthetic-only validation.** No real multi-view dataset has been ingested.
  When one is exported in the `docs/usage-multiview-fusion.md` format, the
  follow-up is small: point a config (with a `fusion` block) at the new `.pkl`
  - no pipeline code changes should be required. This is the exact follow-up
  the acceptance checklist asks to be named.
- **`--grid` does not sweep fusion hyperparameters** (`CCA_GRID`/`GFCCA_GRID`)
  for multi-view cohorts - an intentional, documented scope limit (Step 3.4).
  Sweeping them too is a reasonable follow-up, not silently dropped.
- **Plan 01 has no acceptance report / execution-graph entry** - flagged above
  for the reviewer to close.
- **joblib `DeprecationWarning`** when loading legacy `.pkl` (numpy 2.5
  array-shape) - environmental, pre-existing, not from Plan 02.

## Working-tree state

- Branch `master`, 5 implementation commits + report/graph commits.
- Clean except for untracked agent tooling (`.agents/`, `.claude/`), preserved
  as unrelated work and not committed.
- The pre-existing dirty Plan-02 doc edits (`README.md`,
  `docs/plan/02-*.md`, `experiments/datasets/README.md`,
  `docs/usage-multiview-fusion.md`) were treated as **same-plan work** and
  committed alongside the implementation (they are the finalized contract the
  plan implements).
- The work is entirely on `master`; **no worktree merge or branch switch is
  needed**.
