# Acceptance report: Plan 02 - Multi-view CCA/GFCCA fusion pipeline wiring (review)

- **Plan:** `docs/plan/02-multiview-cca-gfcca-fusion-convention.md`
- **Reviewer:** Claude (independent review, not the implementing agent).
- **Baseline commit:** `8effeaf` (Plan 01, `ACCEPTED`).
- **Implementer's report:** `docs/plan/reports/02-multiview-cca-gfcca-fusion-convention.md`
  (marked `IMPLEMENTED`, pending this review).
- **Conclusion:** **ACCEPTED**, after one required fix applied during review
  (commit `2b8c258`).

## Independent verification performed

Re-ran everything the implementer's report claims, plus code-level reading of
`experiments/multiview.py` against `docs/usage-multiview-fusion.md`:

| Check | Result |
|---|---|
| `uv run pytest tests/ -v` | 55 passed (10 in `test_multiview.py`) |
| `uv run ruff check .` / `ruff format --check .` | clean |
| `uv run mypy src/tbls` | clean (scope unchanged, `experiments/` not covered) |
| `uv build && uvx twine check dist/*` | wheel + sdist, both PASSED |
| Manual: `fuse_views(..., view_groups=None, method="cca", cca_k=5, ...)` on a synthetic 2-view fixture | **Found a real bug** (below). |

## Bug found and fixed during review

`experiments/multiview.py::_validate_view_groups(None, view_names)` defaulted
to `[[name] for name in sorted(view_names)]` — **one singleton group per
view**, meaning every view is passthrough and **no CCA/GFCCA fusion happens
at all**. This silently contradicts `docs/usage-multiview-fusion.md` Section
4 ("By default (`fusion.view_groups` omitted), every view present in the
cohort is fused together as one group") and Plan 02 Step 2's explicit
default (`[[*sorted(X_views.keys())]]`, one group of everything).

This is also the CLI's actual default: `experiments/train.py`'s
`view_groups = fusion_cfg.get("view_groups")` is `None` whenever a config
omits `fusion.view_groups` — the expected common case — so **any user who
didn't explicitly declare fusion groups would silently get plain
concatenation instead of fusion, with no error.**

Not caught by the implementer's own tests because:
- `test_fuse_views_single_group_cca`/`_gfcca` passed `view_groups=None` but
  only asserted `f_train.shape[0]`/train-test shape consistency, never the
  actual fused width (which would have been `8+8=16`, i.e. clearly not
  fused, versus the correct `2*cca_k=10`).
- `test_train_cli_multiview_smoke`, the one true end-to-end test, explicitly
  set `view_groups=[["view_a","view_b"]]`, sidestepping the default path
  entirely — confirmed by checking the implementer's manual verification
  transcript, which also used an explicit `view_groups`.

**Fix (commit `2b8c258`):** `None` → `[sorted(view_names)]` (one group of all
views). Added `test_fuse_views_default_view_groups_fuses_all_views_together`
asserting the actual fused width (`2 * cca_k` for the 2-view fixture, not the
raw `8+8` a passthrough default would produce), applied the same
strengthened assertion to the GFCCA counterpart, and changed
`test_train_cli_multiview_smoke` to omit `view_groups` so the default path is
exercised end-to-end through the CLI, not just via direct `fuse_views` calls.
Re-ran the full suite (55 passed) and rebuilt after the fix.

## Everything else: matches the plan and the spec precisely

- Pkl contract, `load_multiview_cohort` both/neither validation, per-view
  independent `StandardScaler`/feature-selection, SMOTE-family rejection,
  index-only (`oversample`/`undersample` via `sample_indices_`) and
  reference-view (`tomek`) resampling, fusion-group partition validation
  (duplicate/missing/extra), singleton-group passthrough (verified by exact
  column equality, not just "no crash") — all implemented exactly as
  specified in `docs/usage-multiview-fusion.md`.
- `fuse_views`'s `cca`→`tbls.cca.project_cca_features` /
  `gfcca`→`tbls.gfcca.project_cca_features` dispatch is correct and never
  mixed (checked by reading the dispatch table directly, not just trusting
  the report).
- `--fusion` CLI option, `--grid`'s documented model-only scope limit for
  multi-view cohorts, single-view cohorts provably unaffected
  (`tests/test_real_dataset_smoke.py` and the existing real-data CLI run
  both still pass unmodified).
- `experiments/hyperparams.py`'s `CCA_DEFAULTS`/`GFCCA_DEFAULTS` keyword
  names checked directly against `tbls.cca.build_cca_features`/
  `tbls.gfcca.build_gfcca_features` signatures — correct.
- Docs: `docs/usage-experiments-cli.md`'s new section read in full, consistent
  with the (now-fixed) actual behavior; no `.zh-CN.md` file touched, per plan.
- Deviations the implementer flagged (`_run_grid` cohort-tuple signature
  change, `sample_indices_`-based resampling instead of a custom helper, the
  plan's `python -c` verification command needing `sys.path` help) are all
  reasonable, correctly scoped, and don't affect correctness.

## Acceptance checklist (from the plan)

- [x] Implementation matches `docs/usage-multiview-fusion.md` (after the
      `view_groups=None` fix).
- [x] SMOTE-family raises; index-only and reference-view resampling keep
      views aligned.
- [x] `fuse_views` never mixes `tbls.cca`/`tbls.gfcca` projection dispatch.
- [x] Fusion-group partition validation correct; singleton passthrough
      proven by exact column equality.
- [x] Single-view cohorts provably unaffected.
- [x] `CCA_DEFAULTS`/`CCA_GRID`/`GFCCA_DEFAULTS`/`GFCCA_GRID` present with
      verified keyword names.
- [x] English-only docs updated, no `.zh-CN.md` touched.
- [x] Report states synthetic-only validation and names the real-data
      follow-up.

## Remaining known scope limits (intentional, not defects)

- No real multi-view dataset ingested yet (none exists).
- `--grid` does not sweep `CCA_GRID`/`GFCCA_GRID` for multi-view cohorts —
  documented, reasonable follow-up.
