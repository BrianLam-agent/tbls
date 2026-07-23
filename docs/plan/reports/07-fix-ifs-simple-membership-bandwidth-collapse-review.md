# Plan 07 review report — fix `compute_if_scores_simple`'s Gaussian membership bandwidth

- **Decision:** `ACCEPTED`.
- **Target plan:** `docs/plan/07-fix-ifs-simple-membership-bandwidth-collapse.md`
  (node `07-ifs-simple-fix`, no hard predecessors; Plan 07 is a standalone
  correctness bug fix in `src/tbls/_ifs.py`).
- **Implementation report:** `docs/plan/reports/07-fix-ifs-simple-membership-bandwidth-collapse.md`
  (implementer, conclusion `IMPLEMENTED`).
- **Review date:** 2026-07-24 (`pi` agent acting as independent reviewer).
- **Branch:** `master`. **Worktree:** single, current. No branch switch,
  no reset, no stash.

## Commits inspected

Plan 07 implementation (all on `master`, no rebasing observed):

| Hash | Subject | Role |
|------|---------|------|
| `cc111fa` | `docs(plan): Plan 07 - root-caused fix for GFTBLS numerical collapse` | the immutable plan (authored by the same agent that implemented; the plan states the root cause was "root-caused and reproduced by the reviewer before writing this plan") |
| `c1b157a` | `fix(tbls): relativize compute_if_scores_simple's Gaussian membership bandwidth` | Step 1 + lockstep Step-2 test update |
| `62178e1` | `test(tbls): realistic-scale IFS collapse regression + real-data GFTBLS non-degeneracy check` | Steps 3, 4 |
| `b041b7a` | `docs: correct if_sigma parameter description (usage-tbls.md + zh-CN)` | Step 5 + Plan-03 workaround-comment refresh |
| `f961aae` | `docs(plan): Plan 07 acceptance report + execution graph (IMPLEMENTED)` | implementer report + graph flip |

Baseline for the review diff: `cc111fa` (plan landed, code untouched). Reviewer
re-verified the diff `cc111fa → f961aae` on `src/tbls/_ifs.py`,
`tests/test_shared_modules.py`, `tests/test_tbls_gftbls_real_dataset.py`,
`docs/usage-tbls.md`, `docs/usage-tbls.zh-CN.md`, `examples/01_train_tbls.py`.

## Scope and compounding obligations

`AGENTS.md` scope discipline: Plan 07 is a `src/tbls/` change (the published
package), so it must keep the sklearn estimator contract intact. Verified: no
`TBLS`/`GraphFuzzyKCCA` constructor signature, parameter, or default changed;
`get_params`/`set_params`/`fit`/`predict`/`predict_proba` untouched. The fix is
an internal numerics correction in one helper with an unchanged signature —
within scope. The plan's Non-goals (don't touch `compute_if_scores_geib`,
don't add a graph-Laplacian eigenvalue floor, don't retune
`discriminative_beta`/`graph_gamma`, don't change `min_weight`) were honored
(NOT_APPLICABLE → no leakage).

## Adversarial verification (independent probes, not the report's numbers)

### Probe 1 — re-run the plan's exact verification commands

| Command | Exit | Observed (re-run by reviewer) |
|---------|------|-------------------------------|
| `uv run --group experiments pytest tests/ -q` | 0 | `78 passed, 26 warnings in 5.78s` (matches report's 78; warnings are pre-existing joblib/numpy 2.5 deprecations + one `build_graph_laplacian` zeros-divide RuntimeWarning, all pre-Plan-07). |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run ruff format --check .` | 0 | `45 files already formatted` |
| `uv run mypy src/tbls` | 0 | `Success: no issues found in 19 source files` |
| `uv build` + `uvx twine check dist/*` | 0 | built `tbls-0.1.0`; both artifacts `PASSED` |

### Probe 2 — Plan-07-specific tests in isolation (no spuritious skip)

`urm --group experiments pytest tests/test_shared_modules.py::test_compute_if_scores_simple_non_degenerate_on_realistic_scale tests/test_tbls_gftbls_real_dataset.py -v` →
`2 passed, 4 warnings in 1.69s`. (The real-data test is NOT skipped on this
dev machine — the pkl is present, so the acceptance-gate code path actually
executed.) Discriminating power of each new test, assessed independently:

- **`test_compute_if_scores_simple_non_degenerate_on_realistic_scale`**: this
  would fail for a plausible *wrong* fix — it inline-reimplements the OLD
  absolute-sigma `mu` and asserts `mu_old.max() < 1e-6` (self-documenting
  negative check reproducing the underflow), then asserts the *fixed*
  function's `s` has `s.std() > 1e-3`, `s.max() > 0.5`, and **not** all at the
  `1e-4` floor. A still-underflowing fix (e.g. a relativization that used the
  wrong scale, or forgot the `median_dist` factor) would produce `s.std()~0` /
  `s.max()~1e-4` and fail. This asserts intended semantics, not the
  implementation.
- **`test_gftbls_does_not_collapse_on_dm_cohort`**: asserts
  `len(unique_pred) > 1` AND `balanced_accuracy > 0.6` for the exact
  `(use_if_weights=True, graph_gamma=0.1)` GFTBLS configuration. The
  threshold (0.6) is clearly above the degenerate 0.5 single-class floor with
  margin; a regression that reintroduces the collapse fails both halves.
  Skips gracefully (not fails) when the pkl is absent. Correct acceptance-gate.

### Probe 3 — independent IFS-layer + end-to-end before/after reproduction on real data

Reviewer's own inline probe (NOT the implementer's reported numbers), on
`experiments/datasets/biomedical_larger.pkl` `"DM"`, plan's exact recipe
(`_extract_xy`, `train_test_split(test_size=0.2, random_state=0, stratify=y)`,
`StandardScaler` fit on train only):

```
median pairwise distance = 18.058
OLD absolute-sigma mu: mean=7.060e-11 max=5.647e-09   (the root-cause underflow)
NEW returned s: min=8.346e-04 max=0.943 std=1.484e-01 frac_at_floor(1e-4)=0.0000
GFTBLS (True,0.1): unique_pred=[0, 1] balanced_acc=0.9237  (threshold>0.6) -> ACCEPT: True
  (False,0.0): unique=[0, 1] bal_acc=0.9237
  (True, 0.0): unique=[0, 1] bal_acc=0.9237
  (False,0.1): unique=[0, 1] bal_acc=0.9237
```

This independently reproduces the plan's "Why" section's root-cause chain
(absolute `sigma_if=1.0` → `mu` underflows to ~1e-11 on a ~18-median-distance
dataset → all `s` at the `1e-4` floor before fix; `0/2989` at floor after fix)
and the implementer's post-fix numbers (`balanced_acc=0.9237`, dual-class on
the GFTBLS combination). The exact failure mode `balanced_accuracy=0.5000,
unique_pred=[0]` from the plan's reproduction is resolved.

### Probe 4 — Non-goal leakage: `compute_if_scores_geib` byte-identical pre/post

`diff <(git show cc111fa:src/tbls/_ifs.py | sed -n '/def compute_if_scores_geib/,/return np.diag(scores)/p') <(sed -n '/def compute_if_scores_geib/,/return np.diag(scores)/p' src/tbls/_ifs.py)` → identical (GEIB BYTE-IDENTICAL: yes). `build_discriminative_graph_laplacian` and `TBLS._solve_weights` untouched (confirmed via `git show --stat c1b157a` limited to `src/tbls/_ifs.py`). Plan's Non-goals honored.

### Probe 5 — other caller of `compute_if_scores_simple` does not regress

`GraphFuzzyKCCA` also dispatches to `compute_if_scores_simple`. Re-ran
`pytest tests/test_gfcca.py tests/test_cca.py -q` → `2 passed` / `2 passed`.
The relativization changes the absolute IFS weights `GraphFuzzyKCCA` receives
(intended — it was the bug), and its tests stay green.

### Probe 6 — `examples/01_train_tbls.py` (Plan-03 workaround-comment refresh)

Re-ran: `Worked example 01: single TBLS run (cohort=DM) ... accuracy=0.9150
balanced_accuracy=0.9119 macro_f1=0.8917` — non-degenerate, matching the
implementer's reported numbers exactly. The Plan-03 stale workaround
comment (describing the `use_if_weights=True` + `graph_gamma=0.1` collapse as
"not fixed here … out of scope") is now accurately refreshed to "has been
fixed upstream (Plan 07)" (verified via the `b041b7a` examples/01 hunk). The
example's `TBLS(...)` call is unchanged (still `graph_gamma=0.0` default to
isolate the IFS-only effect, per the example's own design choice).

## Findings (ordered by severity)

1. **[PASS, non-blocking risk] Degenerate-input edge case: all-identical
   samples produce `nan` in `s` via `0/0`.** With all rows identical, the
   median pairwise distance is exactly 0, so `sigma_eff = sigma_if * 0 = 0`
   and `mu[i] = exp(-dist**2 / (2 * 0))` = `exp(-0/0)` = `nan` (confirmed:
   `RuntimeWarning: invalid value encountered in scalar divide`, `np.isnan(s).any()=True`).
   **However this is a pre-existing latent pattern, not a Plan-07
   regression**: the `delta_if` branch in the *same* function already had
   `threshold = median_dist * delta_if`, which is also `0` for all-identical
   input (degenerate `rho` — no neighbors within a `0` radius), so
   `compute_if_scores_simple` was never meaningful for degenerate-median input.
   Plan 07's Non-goals explicitly defer such hardening ("don't silently chase
   [further instability] inside this plan … treat any further hardening … as
   a separate, not-yet-justified follow-up"). A correct future hardening pass
   would floor `median_dist` for **both** `sigma_if` and `delta_if` (covering
   the pre-existing latent too), which extends beyond Plan 07's literal
   scope — left for a separate follow-up only if a real dataset resurfaces it.
   n=1 single-sample is benign (the `off_diag.size>0 else 1.0` fallback gives
   `median_dist=1.0`, `mu=exp(0)=1`). Realistic datasets never have
   all-identical rows; the acceptance gate (real data, median-dist ~18) is
   unaffected. Not a `FAIL` and not a reviewer direct-fix (fixing it would
   contradict the plan's deliberate scoping and touch the pre-existing
   `delta_if` latent pattern too).
2. **[PASS, noted deviation] Step-2 test update bundled with the fix commit**
   (implementer deviation #1). The plan suggested Step-2 (the
   `test_compute_if_scores_simple_vectorized_matches_loop` lockstep update) as
   its own follow-up to commit 1, but the implementer bundled it into
   `c1b157a`. Verified reasoning: without the lockstep update, `c1b157a` alone
   would leave that existing `atol=1e-12` regression **red** (its inline "old
   loop" still uses the buggy absolute-sigma formula while the function uses
   `sigma_eff = sigma_if * median_dist`), breaking `git bisect`. Bundling
   keeps every commit bisect-clean. The deviation is disclosed in the report
   and is a bisect-cleanliness improvement, not a scope violation.
3. **[PASS, noted non-action] `_ifs.py` module docstring left stale.** The
   module docstring claims `compute_if_scores_simple` is "used by
   `tbls.gfcca.GraphFuzzyKCCA`" — true, but incomplete (it is also, and
   primarily now under Plan 01's default `if_strategy="simple"`, used by
   `TBLS`). Plan 07's Non-goals scope the change to the
   `compute_if_scores_simple` **formula**, not module-docstring prose.
   Deliberately untouched and flagged by the implementer; cosmetic
   maintainability item for a future pass. Not a `FAIL`.
4. **[PASS] Implementer's "reproduced before the fix" claim.** The plan's
   "Why" table and the implementer's "Pre-fix regression evidence"
   reproduce the same root-cause failure mode (the absolute-magnitude
   difference in the non-collapse rows — plan 0.7279 vs implementer 0.9237 —
   is because `_extract_xy` binarizes `"DM"` to a smaller binary task than
   the plan's external reproduction; the implementer discloses this). The
   acceptance-relevant evidence — the `(True, 0.1)` GFTBLS collapse to
   `0.5000/single-class` and its resolution after the fix — is faithfully
   reproduced by both the implementer's table and the reviewer's independent
   probe. The collapse root cause is genuinely fixed, not papered over.

## Acceptance traceability matrix

| Plan item | Governing | File/commit | Independent probe | Status |
|-----------|-----------|-------------|-------------------|--------|
| Step 1: `mu` uses `sigma_if * median_dist` (relative) | Plan §Step 1; AGENTS scope discipline | `src/tbls/_ifs.py` @ `c1b157a` | Reviewer diff re-read; `compute_if_scores_geib` byte-identical check | **PASS** |
| Step 2: vectorization regression test updated to use corrected formula | Plan §Step 2 | `tests/test_shared_modules.py::test_compute_if_scores_simple_vectorized_matches_loop` @ `c1b157a` | `pytest` of that fn PASSED | **PASS** |
| Step 3: realistic-scale non-degeneracy test asserts old `mu` underflow + new `s` spread | Plan §Step 3, Acceptance checklist | `test_compute_if_scores_simple_non_degenerate_on_realistic_scale` @ `62178e1` | Isolation `pytest` PASSES; discriminating-power assessment PASS | **PASS** |
| Step 4: real-data GFTBLS gate `balanced_accuracy > 0.6`, skips if pkl absent | Plan §Step 4 (primary acceptance gate) | `tests/test_tbls_gftbls_real_dataset.py::test_gftbls_does_not_collapse_on_dm_cohort` @ `62178e1` | Reviewer's independent end-to-end probe: `unique_pred=[0,1]`, `bal_acc=0.9237 > 0.6` | **PASS** |
| `compute_if_scores_geib` unchanged | Plan Non-goal | `src/tbls/_ifs.py` | `diff` byte-identical pre/post | **PASS** |
| Step 5: `if_sigma` doc corrected (English + zh-CN) | Plan §Step 5; AGENTS English-source-of-truth convention | `docs/usage-tbls.md`, `docs/usage-tbls.zh-CN.md` @ `b041b7a` | `git show` hunks reviewed | **PASS** |
| Plan-03 workaround comment updated/stale-removed | Plan Acceptance checklist | `examples/01_train_tbls.py` @ `b041b7a` | `git show` hunk + re-run `01` non-degenerate | **PASS** |
| No estimator-contract / signature / default regression | AGENTS §"sklearn-compatible" | `src/tbls/tbls.py`, `gfcca.py` (untouched) | `git show --stat` confirms scope-limited to `_ifs.py` | **PASS** |
| Full verification gate (pytest/ruff/format/mypy/build/twine) | Plan §Verification | all commits | reviewer re-ran all, exit 0 | **PASS** |

Every hard acceptance item is `PASS`. No `FAIL`. No `UNVERIFIED`.

## Direct reviewer fixes

**None.** No defect met the "fix directly during review" criteria:

- The all-identical `nan` edge case (finding 1) is a pre-existing latent
  pattern covering both `sigma_if` and `delta_if`, explicitly deferred by the
  plan's Non-goals; correcting it would widen scope into
  past-the-single-root-cause hardening the plan deliberately scoped out, and
  would touch the pre-existing `delta_if` latent too. It is not a `FAIL` of a
  Plan-07 contract. Reviewer direct-fix is therefore inappropriate; recorded
  as a non-blocking risk for a future hardening pass.
- The Step-2 bundling (finding 2) and the stale module docstring (finding 3)
  are disclosed non-actions / deviations, not defects.

## Passed / failed / skipped checks

- **Passed:** all six plan-verification commands (exit 0); the two Plan-07
  regression tests; `test_gfcca`/`test_cca` (other IFS caller); the full
  `tests/` suite (78); the reviewer's independent IFS-layer + end-to-end
  real-data probe; `examples/01_train_tbls.py` end-to-end; byte-identical geib
  check; build + twine.
- **Failed:** none.
- **Skipped:** none on this dev machine (the pkl is present, so the real-data
  gate ran rather than skipped). In CI the real-data test skips gracefully —
  the realistic-scale synthetic test (which always runs) is the
  always-on regression guard.
- **Timed-out / unavailable:** none.

## Security / data handling

Plan 07 touches no data pipeline, secret, credential, dataset file, or
network/IO boundary. The IFS computation reads only the in-memory feature
matrix `A` and labels `y` it is handed; `cdist(A, A)` stays in-process. No
restricted data was staged, committed, printed, or copied. The
`experiments/datasets/*.pkl` exercised by the real-data test is the existing
git-ignored dev fixture (not committed). No secrets involved.

## Downstream effect

Plan 07 declares **no downstream nodes** (it is a standalone correctness
fix). Node `03-`/`04-`/`05-`/`06-` are already `ACCEPTED`; node `07` had no
hard predecessors and no dependents. Therefore: accepting Plan 07 releases
**nothing new** (nothing was blocked on it) but closes the GFTBLS-numerical-
collapse finding that Plan 03's review and Plan 04's ablation work had flagged
separately. The found follow-ups the plan explicitly deferred
(`build_discriminative_graph_laplacian` eigenvalue-floor hardening,
`discriminative_beta`/`graph_gamma` re-tuning, and the all-identical-input
`median_dist=0` floor hardening — finding 1) all remain their own potential
future plans if and only if a real dataset resurfaces instability with this
fix already in place; none is required for Plan 07's acceptance.

## Remaining risks and required user/external actions

- **Non-blocking risk (finding 1):** `compute_if_scores_simple` produces `nan`
  for all-identical input (`median_dist=0` → `0/0` in `mu`). Pre-existing latent
  for `delta_if` too, pathologically only for degenerate input, explicitly
  deferred by Plan 07's Non-goals. Future hardening: floor `median_dist` (e.g.
  `max(median_dist, tiny)`) consistently for *both* `sigma_if` and `delta_if`;
  a follow-up plan, not this review.
- **Cosmetic non-blocking (finding 3):** `_ifs.py` module docstring's
  "used by `GraphFuzzyKCCA`" prose is incomplete (also-used-by-`TBLS`-default).
  Future maintainability pass.
- **No user/external actions required** to accept Plan 07. Reviewer acceptance
  completes the contract.

## Final decision

**ACCEPTED.** Every hard acceptance item in the plan's checklist is `PASS`;
the primary acceptance gate (GFTBLS no longer collapses on the real `"DM"`
cohort, `balanced_accuracy > 0.6`) is independently reproduced
(`balanced_accuracy = 0.9237`, dual-class); the root cause is genuinely fixed
(not papered over — confirmed at the IFS layer by the
old-vs-new-`mu` inline probe and by the vectorization regression test staying
bit-for-bit consistent); the fix is correctly scoped (no estimator-contract /
signature / default regression; `compute_if_scores_geib` byte-identical;
plan's Non-goals honored); and all plan + repo gates are green. The two noted
deviations are disclosed and improve bisect-cleanliness; the one non-blocking
risk is correctly deferred per the plan's own Non-goals.

The implementer's report is accurate (its numbers were reproduced by the
reviewer's independent probe to the decimal). The implementing agent does not
grant `ACCEPTED`; this review does.

## Working-tree state and preserved changes

Review performed in the current workspace on `master`; no worktree, no branch
switch, no reset/restore/stash/clean used. Working tree was clean at review
start and contains, besides the two files this review adds (this report +
the `07` graph flip), nothing. No concurrent-agent commits raced the index
during the Plan-07 review window; the most recent commits
(`f7f6bf5` reviewer-acceptance of 03–06; `c8fae94` a reviewer fix to my
earlier `visualize.py` npz depth assumption found during that review; then
the Plan-07 chain) were all already on `master` and stable before this review
began. Everything — implementation, tests, docs, implementer report, and this
review report + graph flip — is on the current `master` branch; no worktree
merge is required of the user.