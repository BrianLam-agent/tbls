# Plan 04: GTBLS/FTBLS/GFTBLS ablation convenience + vectorize graph/IFS hot loops

> Status: final, ready to hand off. **Hard predecessor: none new** (builds on
> the already-`ACCEPTED` graph/IFS strategy switch from Plan 01), but touches
> the same files (`src/tbls/_graph.py`, `src/tbls/_ifs.py`, `src/tbls/gfcca.py`,
> `src/tbls/tbls.py`) so should not run concurrently with another plan
> touching those files.

## Goal

Two independent but co-located changes to the same files:

1. **Ablation ergonomics**: a convenience factory that builds "GTBLS"
   (graph-only), "FTBLS" (fuzzy/IFS-only), "GFTBLS" (both), or plain "TBLS"
   (neither) without adding a new, potentially-conflicting constructor
   parameter — answering the collaborator's ask for an ablation-experiment
   switch.
2. **Vectorize the O(n) / O(n²) pure-Python loops** in `_graph.py`, `_ifs.py`,
   and `gfcca.py` with numpy broadcasting, with **bit-for-bit** regression
   tests proving no behavior change. This is a performance change, not a
   correctness change — see "Why not Cython/pybind11" below.

## Why (context)

TBLS already has two **independent** boolean/float switches that jointly
express the ablation axes: `use_if_weights: bool` (the "F" switch) and
`graph_gamma: float` (the "G" switch — `0` disables it). All four ablation
combinations are already reachable:

| Variant | `use_if_weights` | `graph_gamma` |
|---|---|---|
| TBLS (neither) | `False` | `0.0` |
| GTBLS (graph only) | `False` | `> 0` |
| FTBLS (fuzzy only) | `True` | `0.0` |
| GFTBLS (both — today's tuned default combination) | `True` | `> 0` |

Adding a third `variant` constructor parameter that *also* sets these two
would create two ways to express the same state (risk: `variant="ftbls"`
plus an explicit conflicting `graph_gamma=0.5` — which wins?) and complicate
`sklearn.base.clone()`/`get_params()` round-tripping. A **factory function**
avoids this: it only ever produces `TBLS` instances via the existing,
already-correct constructor, so there is exactly one source of truth.

Separately: `_graph.py::build_discriminative_graph_laplacian` (TBLS's
**default** graph strategy since Plan 01, so it runs on every `fit()` unless
`graph_strategy="knn"` is chosen) contains a pure-Python `for i in range(n):
for j in range(n)` adjacency loop — O(n²) in Python, not numpy, the classic
slow pattern. `_ifs.py::compute_if_scores_geib`/`compute_if_scores_simple`
each have an O(n) neighbor-averaging loop (already flagged as a "Cython
acceleration candidate" in `_ifs.py`'s own module docstring).
`gfcca.py::GraphFuzzyKCCA._build_discriminative_graph` has the same nested
loop as the original this was ported from.

## Why not Cython/pybind11 (decision, not open question)

All of the loops above are same-class/different-class adjacency and
neighbor-averaging patterns that vectorize cleanly and exactly with numpy
broadcasting (e.g. `same = (y[:, None] == y[None, :])`,
`rho = (mask & diff_class).sum(axis=1) / mask.sum(axis=1)`) — this exact
vectorization was already proven bit-for-bit equivalent to the loop version
in `tests/test_shared_modules.py::test_build_discriminative_graph_laplacian_matches_gfcca_reference`'s
independent reference implementation. Vectorizing gets the same order-of-magnitude
speedup as a compiled extension would for this kind of embarrassingly
parallel elementwise/reduction work, with none of a compiled extension's
costs (per-platform wheel builds, a C/C++ or pybind11 toolchain requirement
for contributors, harder-to-review diffs, harder-to-debug numerical issues).
Decision trees (the other computationally heavy part of TBLS/BLS) already
delegate to `sklearn.tree.DecisionTreeRegressor`, itself Cython-compiled —
no action needed there. **Do not introduce a compiled extension in this
plan.** If real profiling against actual production-scale data later shows
these vectorized versions are still a bottleneck, that's a future decision
with real evidence behind it, not a speculative one now.

## Design references

- `docs/usage-tbls.md` — gets a new "Ablation variants" section.
- `docs/architecture.md` section 4 (shared modules table) — update the
  `_graph`/`_ifs` row descriptions if their implementation strategy changes
  materially (it shouldn't from a caller's perspective — same output, faster).
- `tests/test_shared_modules.py::test_build_discriminative_graph_laplacian_matches_gfcca_reference` —
  the existing bit-for-bit pattern to follow for the new vectorized-vs-loop
  regression tests.

## Non-goals

- Any new `TBLS` constructor parameter.
- Changing `graph_strategy="knn"`/`if_strategy="geib"` (the legacy paths) —
  only the **default** strategy's implementations
  (`build_discriminative_graph_laplacian`, `compute_if_scores_simple`, and
  `compute_if_scores_geib`'s neighbor loop) are in scope for vectorization.
  Vectorize `compute_if_scores_geib` too since it's explicitly flagged as a
  candidate in its own docstring, but its behavior must not change either.
- Any change to the *numerical output* of any function — every change here
  must be provably bit-for-bit identical (`np.allclose` with a tight
  tolerance, ideally exact equality where floating-point order of operations
  allows) to the current loop-based output.
- Compiled extensions (Cython/pybind11) — see decision above.

## Implementation steps

### Step 1 — Ablation factory: `tbls.tbls.build_tbls_variant`

```python
def build_tbls_variant(
    variant: Literal["tbls", "gtbls", "ftbls", "gftbls"],
    graph_gamma: float = 0.1,
    **kwargs: Any,
) -> TBLS:
    """Build a TBLS configured for one ablation variant.

    TBLS already exposes the graph and fuzzy-IFS terms as two independent
    switches (``use_if_weights: bool``, ``graph_gamma: float`` where ``0``
    disables it) -- see docs/usage-tbls.md. This factory is a convenience for
    ablation studies (GTBLS = graph only, FTBLS = fuzzy/IFS only, GFTBLS =
    both, TBLS = neither); it does not add new state to the estimator, only
    picks the (use_if_weights, graph_gamma) combination for a name.

    Args:
        variant: One of "tbls" (neither), "gtbls" (graph only), "ftbls"
            (fuzzy/IFS only), "gftbls" (both -- today's tuned default
            combination).
        graph_gamma: Graph regularization weight to use when the variant
            enables the graph term ("gtbls"/"gftbls"). Ignored (forced to
            0.0) for "tbls"/"ftbls". Must be > 0 if provided for a
            graph-enabled variant (else the graph term would silently be a
            no-op, defeating the point of the ablation).
        **kwargs: Forwarded to the ``TBLS`` constructor (e.g.
            n_map_trees, graph_strategy, if_strategy, random_state).
            Must not include ``use_if_weights`` or ``graph_gamma`` --
            raises ValueError if either is passed (this factory's whole
            purpose is to set them unambiguously from ``variant``).

    Returns:
        A configured, unfitted ``TBLS`` instance.

    Raises:
        ValueError: Unknown ``variant``, ``graph_gamma <= 0`` for a
            graph-enabled variant, or ``use_if_weights``/``graph_gamma``
            passed in ``kwargs``.
    """
```

Implementation: a lookup table `{"tbls": (False, 0.0), "gtbls": (False,
graph_gamma), "ftbls": (True, 0.0), "gftbls": (True, graph_gamma)}`,
validate `graph_gamma > 0` for the two graph-enabled variants, validate
`kwargs` doesn't already contain the two managed keys, then
`TBLS(use_if_weights=..., graph_gamma=..., **kwargs)`. Export from
`tbls/__init__.py`'s `__all__`.

### Step 2 — Vectorize `_graph.py::build_discriminative_graph_laplacian`

Replace the nested Python loop with the vectorized form already used as the
independent reference in
`tests/test_shared_modules.py::test_build_discriminative_graph_laplacian_matches_gfcca_reference`:

```python
same = (y[:, None] == y[None, :]).astype(np.float64)
np.fill_diagonal(same, 0.0)
diff = 1.0 - same
np.fill_diagonal(diff, 0.0)
ww = (same + same.T) / 2
wb = (diff + diff.T) / 2
```
(then unchanged: `dw`/`db`/`lw`/`lb`/`normalize`/final combination). Since
`same`/`diff` are already symmetric by construction here, the `(x + x.T) / 2`
symmetrization becomes a no-op but keep it for defensive clarity /
consistency with the pre-existing code shape, not as dead code removal in
this pass (a follow-up cleanup, if wanted, is out of scope here).

### Step 3 — Vectorize `gfcca.py::GraphFuzzyKCCA._build_discriminative_graph`

Same transformation as Step 2, applied to the original (this is the function
Step 2's TBLS-side copy was ported from — keep both in sync, they are
intentionally parallel, not deduplicated, per `docs/architecture.md`).

### Step 4 — Vectorize `_ifs.py`'s two neighbor-averaging loops

`compute_if_scores_geib`'s `lambda_[i] = np.mean(y[neighbors] != y[i])` loop
and `compute_if_scores_simple`'s `rho[i] = np.mean(y[neigh] != y[i])` loop:
both reduce to, given a boolean neighbor mask `M` (shape `(n, n)`, `M[i,j]`
true if `j` is a neighbor of `i`, diagonal excluded) and
`diff = (y[:, None] != y[None, :])`:

```python
neighbor_counts = M.sum(axis=1)
mismatch_counts = (M & diff).sum(axis=1)
result = np.divide(
    mismatch_counts, neighbor_counts,
    out=np.zeros(n, dtype=np.float64), where=neighbor_counts > 0,
)
```

(`np.divide(..., where=...)` avoids the loop's implicit "0 if no neighbors"
branch without a `0/0` warning.) Apply this to both functions' respective
neighbor-mask constructions (`kernel_dists[i] <= sigma` for GEIB,
`dists[i] < threshold` for simple) — build the full `(n, n)` mask matrix
directly instead of per-row in a loop.

### Step 5 — Regression tests

For each vectorized function, add a test that:
1. Computes the OLD loop-based result via an inline reimplementation of the
   pre-vectorization loop (copy the removed loop body into the test, exactly
   as `test_build_discriminative_graph_laplacian_matches_gfcca_reference`
   already does for its independent reference) on a fixed-seed synthetic
   dataset.
2. Computes the NEW vectorized function's result.
3. Asserts `np.allclose` (tight tolerance, e.g. `atol=1e-12`) or exact
   equality between the two.

Do this for `build_discriminative_graph_laplacian`,
`GraphFuzzyKCCA._build_discriminative_graph`, `compute_if_scores_geib`, and
`compute_if_scores_simple`.

### Step 6 — Docs

`docs/usage-tbls.md`: new "Ablation variants (GTBLS/FTBLS/GFTBLS)" section
documenting `build_tbls_variant` with the same table as this plan's "Why"
section, plus a short example:

```python
from tbls.tbls import build_tbls_variant

gtbls = build_tbls_variant("gtbls", graph_gamma=0.2, n_map_trees=15)
ftbls = build_tbls_variant("ftbls", n_map_trees=15)
gftbls = build_tbls_variant("gftbls", graph_gamma=0.2, n_map_trees=15)
```

Mention (English + update `.zh-CN.md` — this one is small enough that both
should be done, unlike the multi-view doc which was explicitly English-only
for now — confirm with the user if in doubt rather than guessing).

### Step 7 — Ablation example (ties into Plan 03)

If Plan 03 (`examples/`) has already landed, add
`examples/03_ablation_gtbls_ftbls_gftbls.py`: fits all four variants on the
same real-data train/test split, prints a small comparison table (accuracy,
balanced accuracy, macro F1 per variant). If Plan 03 has not landed yet, add
this as a follow-up note in the acceptance report rather than blocking on it
or duplicating Plan 03's scaffolding.

## Verification commands

```bash
uv run pytest tests/ -v          # includes new vectorization regression tests
uv run ruff check . && uv run ruff format --check .
uv run mypy src/tbls
uv build && uvx twine check dist/*
```

Additionally, a manual timing sanity check (not a hard pass/fail gate, just
evidence in the report) comparing `build_discriminative_graph_laplacian`
before/after on a few hundred to a few thousand synthetic samples, to confirm
the vectorization is actually faster and not a wash at this scale.

## Acceptance checklist

- [ ] `build_tbls_variant` produces exactly the documented
      `(use_if_weights, graph_gamma)` combination for each of the 4 variants;
      raises on unknown variant, non-positive `graph_gamma` for a
      graph-enabled variant, and on `use_if_weights`/`graph_gamma` in
      `**kwargs`.
- [ ] All 4 vectorized functions are proven bit-for-bit (or `atol=1e-12`)
      identical to their pre-vectorization loop behavior via dedicated
      regression tests.
- [ ] No `TBLS`/`GraphFuzzyKCCA` constructor signature changed.
- [ ] `docs/usage-tbls.md` (+ `.zh-CN.md`) updated.
- [ ] Manual timing evidence included in the acceptance report.
- [ ] No compiled extension introduced.

## Suggested commits

1. `feat(tbls): build_tbls_variant ablation factory (GTBLS/FTBLS/GFTBLS)`
2. `perf(tbls): vectorize discriminative-graph adjacency loop (_graph.py, gfcca.py)`
3. `perf(tbls): vectorize IFS neighbor-averaging loops (_ifs.py)`
4. `docs: ablation variants section (usage-tbls.md + zh-CN)`
