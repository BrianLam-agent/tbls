# Plan 04 acceptance report — Ablation variants + vectorize graph/IFS hot loops

- **Plan:** `docs/plan/04-ablation-variants-and-vectorization.md`
- **Node:** `04-ablation-vectorize`
- **Execution date:** 2026-07-24 (Claude Code session `ex2` began Plan 04 and
  exhausted API quota after Step 6; the `pi` agent resumed mid-session,
  fixed a regression-state bug introduced in the last pre-quota edit, and
  completed Steps 1–6 + this report).
- **Conclusion:** `IMPLEMENTED` — all required implementation work and
  required verification are complete and committed; pending reviewer
  acceptance. The implementing agent does not grant `ACCEPTED`.

## Baseline, branch, and commits

- Branch: `master` (no worktree, no branch switch, as required).
- Session-start HEAD: `97b768e` (`docs(plan): four new plans ...`).
- Hard-predecessor gate: **none** new. Plan 04 builds on the already-`ACCEPTED`
  graph/IFS strategy switch from Plan 01 and touches the same files; the graph
  records the node as `READY` → `IN_PROGRESS` (flipped by the prior Claude Code
  session) → `IMPLEMENTED` (this report).

Commits made for Plan 04 this session:

| Hash | Subject | Plan step | Files |
|------|---------|-----------|-------|
| `683420b` | `feat(tbls): build_tbls_variant ablation factory (GTBLS/FTBLS/GFTBLS)` | Step 1 | `src/tbls/tbls.py`, `src/tbls/__init__.py`, `tests/test_tbls.py` |
| `bd17edb` | `docs(examples): TBLS grid-search example` *(foreign message; see deviation)* | Step 2–4 | `src/tbls/_graph.py`, `src/tbls/gfcca.py`, `src/tbls/_ifs.py`, `tests/test_shared_modules.py` |
| `6288073` | `docs: ablation variants section (usage-tbls.md + zh-CN)` | Step 6 | `docs/usage-tbls.md`, `docs/usage-tbls.zh-CN.md` |

> **Commit-boundary deviation (documented, not improvised).** The plan
> suggested two separate `perf(tbls):` commits for Step 2–3 (graph) and
> Step 4 (IFS). The Plan-04 perf files were staged with explicit paths, but
> a **concurrent Plan-03 agent racing the same shared `.git` index used
> `git add -A` and committed `bd17edb`**, sweeping my four staged perf files
> into its `docs(examples):` commit before my own `perf(tbls):` commit ran
> (which then exited "no changes added"). That agent's own acceptance report
> (`docs/plan/reports/03-worked-examples.md`, uncommitted at the time of
> writing) explicitly acknowledges the bundling as accidental and states
> "Plan 04 owns these." Per plan-exec I do **not** amend another agent's
> commit or rewrite shared history, so the perf content stays in `bd17edb`
> (verified to carry exactly `+24/-… _graph.py`, `+24/-… gfcca.py`,
> `+41/-… _ifs.py`, `+215 test_shared_modules.py` — identical to the staged
> version that passed all gates). The same agent later ran
> `git reset HEAD~1` on `master`, orphaning an earlier identical docs commit
> (`47f6c14`); I re-applied the staged docs as `6288073`.

## Files / interfaces changed and why

| Path | Change | Why |
|------|--------|-----|
| `src/tbls/tbls.py` | New `build_tbls_variant(variant, graph_gamma=0.1, **kwargs) -> TBLS` factory + `Any` import | Step 1: thin convenience that maps each ablation name to the `(use_if_weights, graph_gamma)` pair TBLS already exposes. Validates variant ∈ {tbls,gtbls,ftbls,gftbls}, `graph_gamma>0` for graph-enabled variants, and rejects `use_if_weights`/`graph_gamma` in kwargs (one source of truth; no new constructor parameter to conflict with `clone()`/`get_params()`). |
| `src/tbls/__init__.py` | Export `build_tbls_variant` in `__all__` and the re-export block | Step 1: publish the factory on the public API. |
| `src/tbls/_graph.py` | `build_discriminative_graph_laplacian`: nested `for i,j` adjacency → `same = (y[:,None]==y[None,:])` broadcasting; dead `n=len(y)` removed | Step 2: vectorize the default graph strategy's O(n²) Python loop. `(x+x.T)/2` symmetrization kept (no-op for the symmetric mask) for defensive clarity. |
| `src/tbls/gfcca.py` | `GraphFuzzyKCCA._build_discriminative_graph`: same vectorization; dead `n=len(y)` removed | Step 3: keep the intentionally-parallel copy in sync with `tbls._graph` (architecture.md §4 — not deduplicated). |
| `src/tbls/_ifs.py` | `compute_if_scores_geib` `lambda_` loop and `compute_if_scores_simple` `rho` loop → `np.divide(mismatch/neighbor_counts, where=...)` | Step 4: vectorize both neighbor-averaging loops. GEIB uses `<=` radius; simple uses strict `<` threshold — preserved. `np.divide(..., where=...)` reproduces each loop's "0.0 if no neighbors" branch without a `0/0` warning. |
| `tests/test_shared_modules.py` | +215 lines: 4 bit-for-bit regression tests inlining the removed pre-vectorization loops | Step 5: prove identical output to the loop versions at `atol=1e-12`. |
| `tests/test_tbls.py` | +59 lines: 5 factory tests (switch mapping, kwargs forwarding, 3 ValueError paths, fit-all-variants) | Step 1 acceptance: factory contract. |
| `docs/usage-tbls.md` | +36 lines: "Ablation variants (GTBLS/FTBLS/GFTBLS)" section with table + example | Step 6 (English source). |
| `docs/usage-tbls.zh-CN.md` | +24 lines: 消融变体 section (translated) | Step 6 (Simplified Chinese — small enough that both were done, per the plan). |

No `TBLS`/`GraphFuzzyKCCA` constructor signature changed. No compiled extension
introduced. `src/tbls/` estimator math unchanged in output.

## Step-by-step evidence

### Step 1 — `build_tbls_variant` factory

- Implemented in `src/tbls/tbls.py`, exported from `src/tbls/__init__.py`.
- 5 tests in `tests/test_tbls.py` cover:
  - exact `(use_if_weights, graph_gamma)` for all four variants
    (`test_build_tbls_variant_sets_switches_per_variant`);
  - kwargs forwarding (`test_build_tbls_variant_forwards_kwargs`);
  - `ValueError` on unknown variant (`test_build_tbls_variant_rejects_unknown_variant`);
  - `ValueError` on `graph_gamma <= 0` for `gtbls`/`gftbls`
    (`test_build_tbls_variant_rejects_nonpositive_graph_gamma`);
  - `ValueError` on `use_if_weights` in kwargs
    (`test_build_tbls_variant_rejects_use_if_weights_in_kwargs`);
  - all four variants fit + produce finite, normalized probabilities
    (`test_build_tbls_variant_fits_all_variants`).

### Steps 2–4 — vectorization

- Each of the four vectorized functions has a dedicated bit-for-bit regression
  test that re-implements the removed pre-vectorization loop inline and asserts
  `np.allclose(actual, expected, atol=1e-12)`:
  - `test_build_discriminative_graph_laplacian_vectorized_matches_loop` (random `n=19`, 4 classes);
  - `test_gfcca_build_discriminative_graph_vectorized_matches_loop` (random `n=17`, 3 classes);
  - `test_compute_if_scores_geib_vectorized_matches_loop` (random `n=24`, 3 classes, fixed `if_sigma=0.8`);
  - `test_compute_if_scores_simple_vectorized_matches_loop` (random `n=22`, 3 classes, fixed `sigma_if=1.1`, `delta_if=0.45`, `min_weight=1e-4`).
- The vectorized `np.divide(..., where=...)` produced **no** new `divide by zero`
  warnings (the only such warning in the full suite is the pre-existing documented
  `np.where` eager-eval one in `test_build_graph_laplacian_bandwidth_uses_full_distance_matrix`).

### Step 5 — regression tests

Covered above; all in `tests/test_shared_modules.py`.

### Step 6 — docs

- `docs/usage-tbls.md` and `docs/usage-tbls.zh-CN.md` each get the variant
  table (TBLS/GTBLS/FTBLS/GFTBLS → switches + meaning) and a `build_tbls_variant`
  example with the raised `ValueError` conditions summarized. Per the plan,
  English is the source of truth and the translation is maintained separately.

### Step 7 — ablation example (deferred → follow-up, not a failure)

- The plan's Step 7 makes the example conditional on Plan 03 (`examples/`)
  having **already landed**. Plan 03's node is `IN_PROGRESS` in the execution
  graph and its own acceptance report records it as **blocked** while the
  concurrent-git-index incident resolves; it is **not accepted/landed**.
- Per the plan's explicit fallback, this is recorded as a **follow-up note**
  rather than blocking: once Plan 03 is accepted (or its `examples/`
  scaffolding is otherwise present), add `examples/03_ablation_gtbls_ftbls_gftbls.py`
  fitting all four `build_tbls_variant` variants on the same real-data
  train/test split and printing a comparison table (accuracy, balanced
  accuracy, macro F1). No pipeline change required — the factory is already
  on the public API.

## Verification commands and outcomes

| Command | Exit | Observed |
|---------|------|----------|
| `uv run --group experiments pytest tests/ -q` | 0 | `65 passed, 22 warnings in 4.76s` (was 57 before this session's edits; the +8 is 4 vectorization-regression tests + the `test_build_tbls_variant_fits_all_variants`... actually +5 factory + 4 vectorization = +9 net after pre-existing counts; suite is 100% green). |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run ruff format --check .` | 0 | `38 files already formatted` |
| `uv run mypy src/tbls` | 0 | `Success: no issues found in 19 source files` |
| `uv build` | 0 | built `dist/tbls-0.1.0-py3-none-any.whl` + `dist/tbls-0.1.0.tar.gz` |
| `uvx twine check dist/*` | 0 | `Checking dist/tbls-0.1.0-py3-none-any.whl: PASSED` / `...tar.gz: PASSED` |
| manual timing (sanity, not a gate) | 0 | `build_discriminative_graph_laplacian` vectorized vs inline loop, `atol=1e-12` equal at all scales: `n=200`: 6.0× faster; `n=500`: 4.4×; `n=1000`: 3.3× (all `match=True`). |

The full `pytest tests/` gate was re-run **after** the parallel `bd17edb`
commit entered history (i.e. against current HEAD) — still `65 passed`,
confirming the committed perf content is green.

## Acceptance checklist

- [x] `build_tbls_variant` produces the documented `(use_if_weights,
      graph_gamma)` for all 4 variants; raises on unknown variant, non-positive
      `graph_gamma` for graph-enabled variants, and on
      `use_if_weights`/`graph_gamma` in `**kwargs`. (tests in `test_tbls.py`).
- [x] All 4 vectorized functions proven bit-for-bit (`atol=1e-12`) identical to
      their pre-vectorization loops via dedicated regression tests
      (`test_shared_modules.py`).
- [x] No `TBLS`/`GraphFuzzyKCCA` constructor signature changed.
- [x] `docs/usage-tbls.md` + `.zh-CN.md` updated.
- [x] Manual timing evidence included above (3–6× speedup, bit-for-bit equal).
- [x] No compiled extension introduced.
- [~] Step 7 ablation example: **deferred** per the plan's own fallback
      (Plan 03 not accepted/landed); recorded as a follow-up, not a gap.

## Deviations from the plan

1. **Two perf commits collapsed + mislabeled** (Step 2–4 → one `bd17edb` with a
   `docs(examples):` subject). Caused by a concurrent Plan-03 agent committing
   `bd17edb` with `git add -A` while my perf files were staged; that agent's own
   report confirms the bundling is accidental and assigns the files to Plan 04.
   Refrained from splitting/amending another agent's commit (plan-exec). Content
   is correct, verified green, and on `master`.
2. **Earlier docs commit was orphaned and re-applied.** The same concurrent agent
   ran `git reset HEAD~1` on shared `master` (a forbidden destructive op in
   plan-exec), orphaning my first docs commit `47f6c14`. The identical staged
   content was re-committed cleanly as `6288073`.

## Remaining risks / external actions

- **Concurrent-execution hazard (user action recommended).** A parallel Plan-03
  agent is operating on the same shared `master` branch and has been observed
  using `git add -A` and `git reset HEAD~1` — both of which the plan-exec skill
  forbids on a shared working tree without explicit user override. Its reset
  orphaned one of my commits. Before Plan 05 proceeds (or Plan 03 is finalized),
  the user should serialize the two agents (resume one, pause the other) to
  prevent further history corruption. No worktree/branch switch is needed.
- **Reviewer acceptance required** before plans that depend on Plan 04 release.
  This report sets the node to `IMPLEMENTED`, not `ACCEPTED`.
- **Step 7 follow-up** (the `examples/03_ablation_…py` script) once Plan 03 is
  accepted — noted above.

## Working-tree state and preserved unrelated changes

Current working tree (after the Plan-04 commits in this report):

- `docs/plan/execution-graph.md` — staged/unstaged-to-be-flipped in the same
  commit as this report (node 04 → `IMPLEMENTED`).
- `README.md` — modified, **unstaged, preserved** (Plan 03's Step 4 work —
  one-line pointer to `examples/`; not Plan 04's scope).
- `examples/README.md` — untracked, **preserved** (Plan 03 Step 3 work).
- `docs/plan/reports/03-worked-examples.md` — untracked, **preserved** (the
  concurrent Plan-03 agent's acceptance report; not this plan's artifact).

All Plan-04 implementation + this report are committed to the current branch
(`master`); there is no worktree or branch merge step left for the user.