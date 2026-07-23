# Plan 07: fix `compute_if_scores_simple`'s non-relative Gaussian membership bandwidth

> Status: final, ready to hand off. No hard predecessor. This is a
> **correctness bug fix** in `src/tbls/_ifs.py`, root-caused and reproduced
> by the reviewer before writing this plan — implement exactly the fix
> below, do not re-investigate from scratch.

## Goal

Fix `compute_if_scores_simple`'s Gaussian membership term (`mu`), which
currently treats `sigma_if` as an **absolute** distance unit instead of
**relative to the data's own distance scale** — unlike the same function's
`delta_if` (already correctly relative: `threshold = median_dist * delta_if`)
and unlike `compute_if_scores_geib`'s `sigma` (already correctly relative:
`sigma = if_sigma * median_dist`). This inconsistency causes `mu` to
numerically underflow to ~0 for essentially every sample on real,
non-toy-scale data, which cascades into a degenerate ridge-regression solve
when `TBLS`'s `use_if_weights=True` and `graph_gamma>0` are combined
(GFTBLS) — reproducibly collapsing to all-one-class predictions.

## Why (root cause, already confirmed — do not re-derive)

Reproduced on `experiments/datasets/biomedical_larger.pkl`, cohort `"DM"`,
held-out split (`n_map_trees=10, n_enhance_trees=10, random_state=0`):

```
use_if_weights=False graph_gamma=0.0: acc=0.8904 balanced_acc=0.7279
use_if_weights=True  graph_gamma=0.0: acc=0.8890 balanced_acc=0.7271
use_if_weights=False graph_gamma=0.1: acc=0.8877 balanced_acc=0.7264
use_if_weights=True  graph_gamma=0.1: acc=0.8864 balanced_acc=0.5000  <- degenerate, single class
```

Root cause chain, traced step by step:

1. `compute_if_scores_simple(A, y, sigma_if=1.0, ...)` computes
   `mu[i] = exp(-(||A[i] - center(y[i])||)**2 / (2 * sigma_if**2))` — using
   `sigma_if` **directly as an absolute Euclidean-distance unit**.
2. On this real, standardized 204-feature dataset, the median pairwise
   distance is ~17.8 (confirmed: `scipy.spatial.distance.cdist` on the
   scaled train split). Any samples' distance to its own class centroid is
   on a comparable order of magnitude — nowhere near `sigma_if=1.0`'s
   implicit assumed scale. `exp(-(~a few)**2 / 2)` underflows to numerical
   zero. Confirmed: `mu` was `~1.8e-13` mean, `2.6e-10` max, across all 2989
   train samples (i.e. **every single sample's membership score is
   numerically zero**).
3. This drives `s` (the returned IFS weight) to collapse to the
   function's own `min_weight` clip (`1e-4`) for **every** sample (confirmed:
   2989/2989 samples exactly at `1e-4`), rather than differentiating
   confident vs. borderline samples as intended.
4. In `TBLS._solve_weights`, `S = diag(s)` being uniformly ~`1e-4 * I`
   uniformly shrinks the data-fit terms (`AᵀSA`, `AᵀSY`) by that same tiny
   factor while `graph_gamma * AᵀLA` (the graph term) is unaffected — with
   `graph_gamma=0` this uniform shrinkage is roughly harmless (confirmed:
   `use_if_weights=True, graph_gamma=0.0` barely differs from the baseline);
   with `graph_gamma>0`, the now-negligible data term lets the graph term's
   structure (the discriminative Laplacian `L = Lw - β·Lb`, which is *not*
   strictly PSD — confirmed one slightly-negative eigenvalue among 2989)
   dominate the solve, producing a degenerate result decoupled from `y`.
5. **Confirmed fix**: relativizing `sigma_if` the same way `delta_if` already
   is in this exact function (`sigma_eff = sigma_if * median_dist`, computed
   from the same `dists`/`median_dist` the function already derives for its
   neighbor threshold) restores a sane `mu` distribution (mean 0.76,
   range 0-0.93 on the same data) and **resolves the collapse**:
   ```
   use_if_weights=False graph_gamma=0.0: balanced_acc=0.7279
   use_if_weights=True  graph_gamma=0.0: balanced_acc=0.7256
   use_if_weights=False graph_gamma=0.1: balanced_acc=0.7264
   use_if_weights=True  graph_gamma=0.1: balanced_acc=0.7308  <- no longer degenerate
   ```

`compute_if_scores_geib` is unaffected (its `sigma` is already relative) —
confirmed by reading its implementation; no change needed there.
`build_discriminative_graph_laplacian`'s mild indefiniteness (one
slightly-negative eigenvalue among 2989, magnitude comparable to the
positive eigenvalues) is very likely **not itself a bug** — a discriminative
graph Laplacian combining a within-class and a subtracted between-class term
is expected to be mildly indefinite by construction; it only becomes
catastrophic when paired with an `S` that has collapsed to near-uniform-zero
via the bug above. Do not "fix" `L`'s indefiniteness in this plan — the
`compute_if_scores_simple` fix above resolves the actual failure; treat any
further hardening of the graph term (e.g. an eigenvalue floor) as a separate,
not-yet-justified follow-up if a future dataset resurfaces instability with
this fix already in place.

## Design references

- `src/tbls/_ifs.py::compute_if_scores_simple` — the function being fixed.
- `src/tbls/_ifs.py::compute_if_scores_geib` — the sibling function that
  already does this correctly (`sigma = if_sigma * median_dist`); match its
  pattern.
- `docs/usage-tbls.md`'s `if_delta`/`if_sigma` parameter table — needs a
  one-line correction once the fix lands (`if_sigma` was previously
  documented ambiguously; state explicitly that it is relative to the
  data's median pairwise distance, matching `if_delta`).
- `tests/test_shared_modules.py::test_compute_if_scores_simple_vectorized_matches_loop` —
  the existing bit-for-bit vectorization regression test from Plan 04; its
  inline "old loop" reference must be updated with the same fix (both sides
  of that comparison must use the corrected formula, or the test would
  compare two different-but-still-buggy implementations against each other
  and miss regressions in the corrected one).

## Non-goals

- Changing `compute_if_scores_geib` (already correct).
- Changing `build_discriminative_graph_laplacian`/`L`'s construction or
  adding an eigenvalue floor (explicitly deferred above — not yet justified).
- Changing `TBLS._solve_weights`'s combination formula.
- Re-tuning `discriminative_beta`/`graph_gamma`/other defaults — the fix is
  scoped to the one root cause identified; if real-data testing after the
  fix reveals a *different* remaining instability, flag it, don't silently
  chase it inside this plan.
- Changing `min_weight`'s default value (`1e-4`) — it was doing its job
  (clipping); the bug was upstream of it (the value it was clipping *to* was
  reached inappropriately, not that the clip itself is wrong).

## Implementation steps

### Step 1 — Fix `compute_if_scores_simple`

In `src/tbls/_ifs.py`, in `compute_if_scores_simple`, move the
`dists`/`median_dist` computation (currently computed only for the `rho`
neighbor-threshold step) earlier, and use it for `mu` too:

```python
def compute_if_scores_simple(
    A: NDArray[np.float64],
    y: NDArray[np.int64],
    sigma_if: float = 1.0,
    delta_if: float = 0.5,
    min_weight: float = 1e-4,
) -> NDArray[np.float64]:
    n = A.shape[0]
    classes = np.unique(y)
    centers = {c: A[y == c].mean(axis=0) for c in classes}

    # Relative distance scale (shared by both the membership term and the
    # neighborhood threshold below) -- computed once, up front.
    dists = cdist(A, A, "euclidean")
    off_diag = dists[~np.eye(n, dtype=bool)]
    median_dist = np.median(off_diag) if off_diag.size > 0 else 1.0

    # Membership: Gaussian in units of the data's own median pairwise
    # distance, matching `delta_if`'s neighborhood threshold below and
    # `compute_if_scores_geib`'s `sigma = if_sigma * median_dist` -- NOT an
    # absolute distance unit (that was the bug: `sigma_if` alone underflows
    # `mu` to numerical zero on any real, non-toy-scale dataset).
    sigma_eff = sigma_if * median_dist
    mu = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ci = y[i]
        dist = np.linalg.norm(A[i] - centers[ci])
        mu[i] = np.exp(-(dist**2) / (2 * sigma_eff**2))
    mu = np.clip(mu, 0.0, 1.0)

    threshold = median_dist * delta_if
    # ... rho computation unchanged, reusing the already-computed `dists` ...
```

Update the docstring's `sigma_if` description to state explicitly it is
relative to the data's median pairwise distance (matching `delta_if`'s
existing docstring language), not an absolute unit. No constructor/function
signature change — same parameters, same defaults, corrected internal
computation only.

### Step 2 — Update the Plan-04 vectorization regression test

`tests/test_shared_modules.py::test_compute_if_scores_simple_vectorized_matches_loop`'s
inline "old loop" reference implementation must apply the same
`sigma_eff = sigma_if * median_dist` correction (compute `median_dist` before
building `mu`, same as Step 1) so the test continues to compare "loop
reimplementation" vs. "current function" on the **same, now-corrected**
formula, not the old buggy one.

### Step 3 — New regression test proving the fix on realistic-scale data

Add a test using synthetic data at a **realistic distance scale** (not the
existing small-magnitude `rng.normal(size=(30, 4))` fixtures, which don't
reproduce the bug) — e.g. scale features so the median pairwise distance is
~15-20 (matching the real dataset's scale that exposed this), with two
separated class clusters. Assert:
- Before-fix behavior would have `mu` collapse near 0 for all samples (this
  can be asserted by literally computing the old formula inline as a
  negative check, or simply asserted as "this test would have failed before
  the fix" in a comment — prefer the inline computation so the test is
  self-documenting and doesn't just assert the fixed function's output in a
  vacuum).
- After the fix, `mu` (or the returned `s` vector) has non-degenerate spread
  (not all values within `1e-6` of `min_weight`).

### Step 4 — Real-data regression test: GFTBLS no longer collapses

Add a test (mark it to skip if the real dataset pkl is absent, same pattern
as `tests/test_real_dataset_smoke.py`) that fits
`TBLS(use_if_weights=True, graph_gamma=0.1, n_map_trees=10,
n_enhance_trees=10, random_state=0)` on `biomedical_larger.pkl`'s `"DM"`
cohort (held-out split, same recipe as this plan's reproduction above) and
asserts `balanced_accuracy_score(y_test, model.predict(X_test)) > 0.6` (a
threshold clearly above the degenerate `0.5`, with margin) — this is the
plan's primary acceptance gate, since it's the exact failure mode being fixed.

### Step 5 — Docs

`docs/usage-tbls.md` (+ `.zh-CN.md`): correct the `if_sigma`/`if_delta`
parameter table row to state both are relative to the data's median pairwise
distance (not just `if_delta`). If the table currently implies `if_sigma` is
an absolute scale anywhere, fix that language. Do not add extended prose
explaining the bug history in the user-facing doc — a short, correct
parameter description is sufficient; this plan's own file is the historical
record.

## Verification commands and test cases

```bash
uv run pytest tests/ -v
uv run ruff check . && uv run ruff format --check .
uv run mypy src/tbls
uv build && uvx twine check dist/*
```

Manual (must reproduce the exact before/after numbers in this plan's "Why"
section, on `experiments/datasets/biomedical_larger.pkl`, cohort `"DM"`):

```python
from tbls import TBLS
# ... load DM cohort, held-out split, StandardScaler, same recipe as above ...
for uw, gg in [(False, 0.0), (True, 0.0), (False, 0.1), (True, 0.1)]:
    m = TBLS(n_map_trees=10, n_enhance_trees=10, use_if_weights=uw, graph_gamma=gg, random_state=0)
    m.fit(x_tr, y_tr)
    # assert balanced_accuracy for (True, 0.1) is no longer ~0.5
```

Also re-run `examples/01_train_tbls.py` and manually try re-enabling
`graph_gamma=0.1` alongside `use_if_weights=True` in that example (the
combination the implementer had to work around per Plan 03's report) to
confirm it's no longer degenerate — if so, note in the acceptance report
that Plan 03's workaround comment is now stale and should be removed/updated
(a small follow-up doc touch, not a new plan).

## Acceptance checklist

- [ ] `compute_if_scores_simple`'s `mu` term uses `sigma_if * median_dist`
      (relative), matching `delta_if` and `compute_if_scores_geib`.
- [ ] `test_compute_if_scores_simple_vectorized_matches_loop` updated
      consistently (both sides of the comparison use the corrected formula).
- [ ] New test proves non-degenerate `mu`/`s` spread on realistic-scale data.
- [ ] New real-data test proves `TBLS(use_if_weights=True, graph_gamma=0.1)`
      no longer collapses on `biomedical_larger.pkl` (`balanced_accuracy >
      0.6`), skipped gracefully if the pkl is absent.
- [ ] `compute_if_scores_geib` unchanged (confirmed already correct).
- [ ] `docs/usage-tbls.md` (+ `.zh-CN.md`) `if_sigma` description corrected.
- [ ] Acceptance report states whether `examples/01_train_tbls.py`'s
      workaround comment (Plan 03) is now stale, and updates it if so.

## Suggested commits

1. `fix(tbls): relativize compute_if_scores_simple's Gaussian membership bandwidth`
2. `test(tbls): realistic-scale IFS collapse regression + real-data GFTBLS non-degeneracy check`
3. `docs: correct if_sigma parameter description (usage-tbls.md + zh-CN)`
