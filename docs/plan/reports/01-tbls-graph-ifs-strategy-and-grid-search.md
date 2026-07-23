# Acceptance report: Plan 01 - TBLS graph/IFS strategy switch and grid search

- **Plan:** `docs/plan/01-tbls-graph-ifs-strategy-and-grid-search.md`
- **Reviewer:** Claude (independent review, not the implementing agent).
- **Implementation commit:** `8effeaf` on `master`.
- **Conclusion:** **ACCEPTED**.

## Why this report is being written after the fact

Plan 01 was implemented and independently verified during the same session as
Plan 02's review, but no acceptance report was written at the time (an
oversight — Plan 02's implementer correctly flagged the gap in its own
report). This report closes that gap using the verification actually
performed at the time, re-confirmed now.

## Summary

`TBLS` gained `graph_strategy` (`"discriminative"` default / `"knn"` legacy)
and `if_strategy` (`"simple"` default / `"geib"` legacy) constructor
parameters. The new defaults port `GraphFuzzyKCCA`'s tuned label-only
discriminative graph Laplacian and per-class-center IFS formula into
`src/tbls/_graph.py`/`_ifs.py`; the legacy kNN-graph/GEIB-IFS behavior is
preserved exactly as an opt-in fallback. `experiments/hyperparams.py`
centralizes `BLS_DEFAULTS`/`BLS_GRID`/`TBLS_DEFAULTS`/`TBLS_GRID` as editable
Python dicts; `experiments/train.py` gained `--model [tbls|bls]` and `--grid`
(full `ParameterGrid` sweep, ranked `GridSummary` Excel sheet).

## Verification performed (independent, this session)

| Check | Result |
|---|---|
| `uv run pytest tests/ -v` | 45 passed |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | all files formatted |
| `uv run mypy src/tbls` | no issues, 19 source files |
| `uv build && uvx twine check dist/*` | wheel + sdist, both PASSED |
| Manual: `TBLS(use_if_weights=True, graph_gamma=0.1)` (new default) vs. `graph_strategy="knn", if_strategy="geib"` (legacy) on real `biomedical_larger.pkl` data, **held-out test split** | Confirmed genuinely different: raw pre-softmax output max abs diff 2.14, `S_`/`L_` matrices substantially different. (A first check evaluated on the training set and found `predict_proba` outputs identical — this was a training-set-saturation artifact of an easily-overfit 8+8-tree model, not a bug; re-checked with a held-out split.) |
| Manual: `experiments/train.py --dataset biomedical_larger --n-splits 2 --grid` | Full 3×3×3 grid ran end-to-end for all 4 cohorts; `GridSummary` sheet ranked descending by `avg_balanced_accuracy`, 27 rows. |
| Manual: `experiments/train.py --dataset biomedical_larger --model bls --n-splits 2` | Ran end-to-end, sane metrics for all 4 cohorts. |
| Real risk found and fixed during review: `BLS/` (346 MB legacy reference tree with duplicate dataset pkls) and root `gfcca.py` were untracked but **not** gitignored — a stray `git add -A` would have committed 346 MB into git history. Fixed in `.gitignore` (and `.ruff.toml` exclude) as part of the same commit. | Fixed, verified with `git check-ignore -v`. |

## Files changed

`src/tbls/_graph.py` (+`build_discriminative_graph_laplacian`), `src/tbls/tbls.py`
(new params + strategy branching in `fit`), `experiments/hyperparams.py` (new),
`experiments/train.py` (`--model`, `--grid`), `tests/conftest.py`,
`tests/test_shared_modules.py`, `tests/test_tbls.py`, `tests/test_experiments_train.py`
(new), `.gitignore`, `.ruff.toml`, `docs/architecture.md`, `docs/usage-tbls.md`,
`docs/usage-experiments-cli.md` (+ their `.zh-CN.md` counterparts).

## Acceptance checklist (from the plan)

- [x] `graph_strategy`/`if_strategy` selectable independently; default
      reproduces `GraphFuzzyKCCA`'s tuned formulas.
- [x] Regression test cross-checks `build_discriminative_graph_laplacian`
      directly against `GraphFuzzyKCCA._build_discriminative_graph` (stronger
      than the plan required — not just an independent reimplementation).
- [x] `graph_strategy="knn"`/`if_strategy="geib"` reproduce pre-plan behavior
      (existing `_graph.py`/`_ifs.py` tests pass unmodified).
- [x] Invalid strategy strings raise `ValueError`.
- [x] `experiments/hyperparams.py` centralizes BLS/TBLS defaults + grids;
      `CCA_*`/`GFCCA_*` left commented (reserved for Plan 02).
- [x] `--model`/`--grid` work end-to-end on real data (manually verified,
      not just unit-tested).
- [x] Docs (English + Chinese) updated.

No open issues. Plan 02 was built on top of this commit without further
changes to it.
