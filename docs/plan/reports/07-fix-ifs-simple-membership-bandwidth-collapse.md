# Plan 07 acceptance report — fix `compute_if_scores_simple` membership bandwidth collapse

- **Plan:** `docs/plan/07-fix-ifs-simple-membership-bandwidth-collapse.md`
- **Node:** `07-ifs-simple-fix`
- **Execution date:** 2026-07-24 (`pi` agent).
- **Conclusion:** `IMPLEMENTED` — all implementation work and required
  verification complete and committed; pending reviewer acceptance. The
  implementing agent does not grant `ACCEPTED`.

## Baseline, branch, and commits

- Branch: `master` (no worktree, no branch switch, as required).
- Session-start HEAD: `cc111fa` (`docs(plan): Plan 07 - root-caused fix ...`).
- Execution-gate check: node `07-ifs-simple-fix` was `READY`, hard predecessors
  **none** (Plan 07 declares none; Plans 03–06 were all `ACCEPTED` as of
  `f7f6bf5`). Gate passed. Working tree was clean at session start.

Commits for Plan 07 this session:

| Hash | Subject | Plan step | Files |
|------|---------|-----------|-------|
| `c1b157a` | `fix(tbls): relativize compute_if_scores_simple's Gaussian membership bandwidth` | 1 (+ lockstep Step-2 update) | `src/tbls/_ifs.py`, `tests/test_shared_modules.py` |
| `62178e1` | `test(tbls): realistic-scale IFS collapse regression + real-data GFTBLS non-degeneracy check` | 3, 4 | `tests/test_shared_modules.py` (new fn), `tests/test_tbls_gftbls_real_dataset.py` (new) |
| `b041b7a` | `docs: correct if_sigma parameter description (usage-tbls.md + zh-CN)` | 5 (+ stale-workaround refresh) | `docs/usage-tbls.md`, `docs/usage-tbls.zh-CN.md`, `examples/01_train_tbls.py` |
| (this commit) | `docs(plan): Plan 07 acceptance report + execution graph (IMPLEMENTED)` | — report | `docs/plan/reports/07-fix-ifs-simple-membership-bandwidth-collapse.md`, `docs/plan/execution-graph.md` |

## Pre-fix regression evidence established independently on this dev machine

Per the skill ("Establish or run the plan's failing regression evidence before
the fix"), the failure mode was reproduced **before any code change** on
`experiments/datasets/biomedical_larger.pkl` (present on this dev machine),
with the plan's stated recipe (`"DM"` cohort, `_extract_xy`, stratified
`train_test_split(test_size=0.2, random_state=0)`, `StandardScaler` fit on
train only, `TBLS(n_map_trees=10, n_enhance_trees=10, random_state=0)`):

| `use_if_weights` | `graph_gamma` | acc | balanced_acc | `unique_pred` |
|:--:|:--:|:--:|:--:|:--:|
| `False` | `0.0` | 0.9267 | 0.9237 | `[0, 1]` |
| `True`  | `0.0` | 0.9267 | 0.9237 | `[0, 1]` |
| `False` | `0.1` | 0.9267 | 0.9237 | `[0, 1]` |
| `True`  | `0.1` | **0.7507** | **0.5000** | **`[0]`** ← degenerate single-class |

This reproduces the plan's exact failure mode (the `(True, 0.1)` GFTBLS
combination collapses to all-one-class predictions). The table's absolute
numbers for the non-collapse rows (0.9237 here vs 0.7279 in the plan) differ
because `_extract_xy` binarizes the DM cohort to a smaller binary task than the
plan's reproduction; the **root-cause failure mode** (degenerate 0.5 on
GFTBLS) is faithfully reproduced and is the acceptance-relevant evidence.

IFS-layer evidence (same recipe, scaled train split, `compute_if_scores_simple`
with `sigma_if=1.0`):

- median pairwise Euclidean distance = **18.06** (plan ≈17.8 ✓)
- returned `s` (IFS weights) before fix: every sample exactly at the
  `1e-4` `min_weight` floor — `frac_at_minweight = 1.0000` (plan: all
  samples at `1e-4` ✓)
- OLD absolute-sigma `mu`: mean `7.06e-11`, max `5.65e-9` (numerically zero;
  plan ≈`1.8e-13` mean ✓)
- NEW relative-sigma `mu` (post fix): mean `0.7695`, max `0.9434`, min
  `8.35e-04` (plan: mean 0.76, range 0–0.93 ✓)

## Files / interfaces changed and why

| Path | Change | Why |
|------|--------|-----|
| `src/tbls/_ifs.py` (`compute_if_scores_simple`) | Move the `cdist`/`median_dist` computation above the `mu` loop; compute `sigma_eff = sigma_if * median_dist` and use it in `mu[i] = exp(-(dist**2)/(2*sigma_eff**2))`; `delta_if`'s `threshold = median_dist * delta_if` unchanged and now reuses the already-computed `dists`. Docstring updated to state `sigma_if` is **relative** to the data's median pairwise Euclidean distance (matching `delta_if`), not an absolute unit. | Step 1: the root-cause fix. No signature/parameter/default change. |
| `tests/test_shared_modules.py` (Step 2) | `test_compute_if_scores_simple_vectorized_matches_loop`'s inline "old loop" reference now computes `median_dist` before `mu` and uses `sigma_eff = sigma_if * median_dist`, so both sides of the `atol=1e-12` comparison apply the corrected formula. New test `test_compute_if_scores_simple_non_degenerate_on_realistic_scale` (Step 3). | Step 2 keeps the bit-for-bit vectorization regression valid after the formula change; Step 3 proves non-degenerate spread on realistic-scale data. |
| `tests/test_tbls_gftbls_real_dataset.py` (new) | Skipped when `biomedical_larger.pkl` absent (`pytest.mark.skipif(not REAL_DATA.exists())`, mirroring `test_real_dataset_smoke.py`); when present, fits `TBLS(use_if_weights=True, graph_gamma=0.1, n_map_trees=10, n_enhance_trees=10, random_state=0)` on the `"DM"` cohort and asserts `len(unique_pred) > 1` and `balanced_accuracy_score > 0.6`. | Step 4: the plan's primary acceptance gate. |
| `docs/usage-tbls.md` / `docs/usage-tbls.zh-CN.md` | Parameter table: `if_sigma` → "Gaussian membership bandwidth, in units of the data's median pairwise Euclidean distance (not an absolute scale; matches `if_delta`)" (was misleadingly described as a "Neighborhood-radius scale"); `if_delta` gains the same "(in units of the data's median pairwise Euclidean distance)" qualifier for symmetry. | Step 5. |
| `examples/01_train_tbls.py` | Comment-only refresh of the Plan-03 workaround comment (the `use_if_weights=True` + `graph_gamma>0` collapse is now fixed upstream; the example's `TBLS(...)` call is unchanged). | Acceptance checklist item: the workaround comment was/is stale. |

`compute_if_scores_geib` is **byte-identical** pre/post fix (`git show
HEAD~2:src/tbls/_ifs.py` vs current — the geib function body up to its
`return np.diag(scores)` is unchanged). `build_discriminative_graph_laplacian`
and `TBLS._solve_weights` untouched. `TBLS`/`GraphFuzzyKCCA` constructor
signatures unchanged. No new dependencies; no compiled extension.

## Step-by-step evidence

### Step 1 — fix `compute_if_scores_simple`

- Verified `TBLS`'s default `if_strategy="simple"` (tbls.py L156 default +
  L316–317 dispatch calls `compute_if_scores_simple`), confirming the plan's
  root-cause attribution (and resolving a stale paragraph in `_ifs.py`'s
  *module* docstring — "simple used by GraphFuzzyKCCA" — which is pre-Plan-01
  prose; **out of this plan's scope** to fix it, so untouched).
- Fix applied exactly as the plan specified: relative `sigma_eff`, `dists`
  computed once up front, `delta_if` reusing it. Post-fix reproduction on the
  DM cohort (same recipe): all four `(use_if_weights, graph_gamma)` combos →
  `acc=0.9267 balanced_acc=0.9237 unique_pred=[0,1]` — **the GFTBLS collapse
  is resolved** (was `0.5000 / [0]`).

### Step 2 — update the Plan-04 vectorization regression test

- `test_compute_if_scores_simple_vectorized_matches_loop` updated in lockstep
  with the fix; re-run in isolation:
  `tests/test_shared_modules.py::test_compute_if_scores_simple_vectorized_matches_loop
  PASSED`.

### Step 3 — realistic-scale non-degeneracy test

- `test_compute_if_scores_simple_non_degenerate_on_realistic_scale` PASSED.
  It (a) sanity-checks the fixture's median pairwise distance is in the
  realistic band (`5 < m < 40`), (b) inline-reimplements the OLD absolute-sigma
  `mu` as a self-documenting negative check (`mu_old.max() < 1e-6` — the
  pre-fix underflow), and (c) asserts the fixed function's `s` has
  non-degenerate spread (`s.min()>0`, not all at the `1e-4` floor,
  `s.std()>1e-3`, `s.max()>0.5`).

### Step 4 — real-data GFTBLS non-degeneracy test

- `test_gftbls_does_not_collapse_on_dm_cohort` PASSED on this dev machine
  (pkl present): predictions span both classes and
  `balanced_accuracy_score ≈ 0.9237 > 0.6`. (CI, where the pkl is absent,
  skips gracefully.)

### Step 5 — docs

- `if_sigma` description corrected in `docs/usage-tbls.md` +
  `docs/usage-tbls.zh-CN.md`; `if_delta` symmetrically qualified. The
  `_ifs.py` *module* docstring's stale "simple used by GraphFuzzyKCCA"
  sentence was deliberately **not** touched (out of scope — Plan 07 is scoped
  to the `compute_if_scores_simple` formula; flagging it here as a deliberate
  non-action, not a gap).

### Acceptance checklist — Plan-03 workaround comment

- `examples/01_train_tbls.py`'s stale workaround comment (describing the
  `use_if_weights=True` + `graph_gamma=0.1` collapse as "not fixed here") was
  **updated** (commit `b041b7a`) to state the collapse is fixed upstream by
  Plan 07 and that the combination now runs non-degenerately. The example's
  `model = TBLS(...)` call is unchanged (it intentionally uses the default
  `graph_gamma=0.0` to isolate the IFS-only effect). Re-running
  `examples/01_train_tbls.py` post-fix: `accuracy=0.9150
  balanced_accuracy=0.9119 macro_f1=0.8917` (non-degenerate).

## Verification commands and outcomes

| Command | Exit | Observed |
|---------|------|----------|
| `uv run --group experiments pytest tests/ -q` | 0 | `78 passed, 26 warnings in 6.43s` (was 76 pre-Plan-07; +2 new tests; **no prior test regressed** — `GraphFuzzyKCCA`, real-dataset smoke, multiview, train, etc. all stay green with the updated IFS numerics). |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run ruff format --check .` | 0 | `45 files already formatted` |
| `uv run mypy src/tbls` | 0 | `Success: no issues found in 19 source files` |
| `uv build` | 0 | built `dist/tbls-0.1.0-py3-none-any.whl` + `dist/tbls-0.1.0.tar.gz` |
| `uvx twine check dist/*` | 0 | both `PASSED` |
| manual reproduction on `biomedical_larger.pkl` `"DM"` (before→after fix) | 0 | `(True, 0.1)`: `0.5000` single-class → `0.9237` dual-class (above). |
| `uv run --group experiments python examples/01_train_tbls.py` | 0 | `acc=0.9150 balanced_accuracy=0.9119 macro_f1=0.8917` (non-degenerate post-fix). |
| `git diff --check` (per-commit) | 0 | clean, no whitespace errors. |

## Deviations from the plan

1. **Commit boundary coupling (Step 2 bundled with the fix).** The plan's
   suggested commit 1 ("fix") was followed, but the Step-2 *update* to the
   existing bit-for-bit vectorization regression test
   (`test_compute_if_scores_simple_vectorized_matches_loop`) was bundled into
   commit `c1b157a` rather than held for commit 2. Rationale: without the
   lockstep test update, commit 1 alone would leave its own existing
   regression test **red** (the inline "old loop" still uses the buggy
   absolute-sigma formula while the fixed function uses `sigma_eff =
   sigma_if * median_dist`), breaking `git bisect`. Bundling keeps every
   commit bisect-clean. The plan's other suggested commits (2 = the two new
   regression tests, 3 = docs) are followed unchanged, via a clean hunk-split
   of `tests/test_shared_modules.py`.
2. **`_ifs.py` module docstring left stale.** The module docstring claims
   `compute_if_scores_simple` is "used by `tbls.gfcca.GraphFuzzyKCCA`" (it is,
   but it's *also* — and primarily now — used by `TBLS` under the default
   `if_strategy="simple"`). Correcting that prose is **not** in Plan 07's
   scope (Non-goals: "Changing `compute_if_scores_geib` (already correct)",
   and the plan touches only `compute_if_scores_simple`'s formula).
   Deliberately left untouched and flagged here for a future maintainability
   pass.

## Remaining risks / external actions

- **Reviewer acceptance required** to move node `07-ifs-simple-fix` from
  `IMPLEMENTED` to `ACCEPTED`. The implementing agent does not grant
  `ACCEPTED`.
- **Other callers of `compute_if_scores_simple`** (`GraphFuzzyKCCA`) now
  receive the relativized version too. All `GraphFuzzyKCCA`-exercising tests
  remain green; the realistic-scale `01_train_tbls.py` example produces
  non-degenerate output. Any downstream external consumers that depended on
  the old absolute-scale semantics (unlikely — that was the bug) would see
  numerically different IFS weights; flagged for the reviewer.
- **Plan-noted follow-up not done here**: the plan explicitly defers any
  eigenvalue-floor hardening of `build_discriminative_graph_laplacian`'s mild
  indefiniteness ("not yet justified") and any re-tuning of
  `discriminative_beta`/`graph_gamma` defaults. Those remain follow-ups if a
  future dataset resurfaces instability with this fix in place.
- The plan's last verification suggestion — manually re-enabling
  `graph_gamma=0.1` in `examples/01_train_tbls.py` — was exercised via the
  standalone reproduction script (the plan's exact recipe), which showed the
  `+IFS +graph` combination now runs at `balanced_acc=0.9237` (above). The
  example file itself was kept at `graph_gamma=0.0` (its scoping choice) with
  only the now-stale caveat comment refreshed.

## Working-tree state and preserved unrelated changes

After the three implementation commits, the working tree holds only the
execution-graph flip (node `07-ifs-simple-fix` → `IMPLEMENTED`), staged/to-be
committed together with this report. No parallel agents were observed writing
to the shared `master` branch during this Plan-07 session (no `git reset` /
`git add -A` cross-lane incidents this window). All Plan-07 implementation +
this report are on the current branch (`master`); there is no worktree or
branch merge step left for the user.