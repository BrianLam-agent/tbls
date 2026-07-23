# Plan 01: TBLS graph/IFS strategy switch, model selection, and grid search

> Status: draft, not yet handed off.

## Goal

1. Give `tbls.TBLS` a switchable graph-regularization strategy and a
   switchable IFS-scoring strategy: the currently-baked-in
   kNN-graph + GEIB-IFS formulas (never tuned) stay available, but the
   default becomes `GraphFuzzyKCCA`'s already-tuned discriminative-graph +
   simple-IFS formulas (see "Why" below — these are byte-for-byte the same
   math already in `tbls.gfcca.GraphFuzzyKCCA`, just not yet reachable from
   `TBLS`).
2. Let `experiments/train.py` train either `TBLS` or `BroadLearningSystem`
   (today only `TBLS` is wired up, even though the config already has an
   unused `model.name` field), with default hyperparameters centralized as
   plain, directly-editable constants — matching the "tune it directly in
   main.py" request.
3. Add a built-in `--grid` grid-search mode to `experiments/train.py` that
   sweeps a small hyperparameter grid (defined next to the same constants)
   per model, runs the existing k-fold CV for every combination, and writes
   a ranked summary on top of the existing per-fold Excel output.

## Why (context for the "which gfcca do we use" question)

The user pasted a root-level `gfcca.py` and asked whether it differs from
`src/tbls/gfcca.py`. Direct comparison (structural diff with comments/vars
normalized, plus a line-by-line check of `_build_discriminative_graph`,
`_compute_if_scores`, and `fit`'s block-matrix/retry logic) shows **no
algorithmic difference** — `src/tbls/gfcca.py::GraphFuzzyKCCA` already *is*
this exact tuned model (same constructor params, same formulas, same retry
logic); the refactor only lowercased local variable names, added type hints,
and delegated the IFS formula to the shared `tbls._ifs.compute_if_scores_simple`
(itself an unmodified port). **No change to `gfcca.py` itself is needed or
proposed by this plan.**

The actual request, reconstructed from "这个模型我调了拉普拉斯图，那个没有调 ...
等于这个的拉普拉斯图和直觉模糊计算，去结合之前 tbls.py 的 TBLS 算法": bring
`GraphFuzzyKCCA`'s **tuned** graph-Laplacian (`Lw - beta * Lb`, built purely
from labels, no kNN) and its **tuned** IFS scoring
(`compute_if_scores_simple`: per-class Euclidean center distance + a
relative-distance neighborhood threshold) into `TBLS`, which currently only
has its own, never-tuned kNN-graph (`_graph.build_graph_laplacian`) and GEIB
IFS (`_ifs.compute_if_scores_geib`). Make the tuned pair the default, keep
the old pair reachable by parameter — exactly the "one parameter, two
options, default new / can switch to old" the user asked for.

## Design references

- [`docs/architecture.md`](../architecture.md) section 4 (shared modules) and
  section 5 (estimator contract) — this plan extends both.
- [`docs/usage-tbls.md`](../usage-tbls.md) — needs a new section for the two
  strategies once implemented.
- [`docs/usage-experiments-cli.md`](../usage-experiments-cli.md) — needs a
  new section for `--grid` and the centralized defaults module.
- `src/tbls/_graph.py`, `src/tbls/_ifs.py`, `src/tbls/gfcca.py`,
  `src/tbls/tbls.py`, `experiments/train.py` — files this plan touches.

## Upstream dependencies

None. First plan since the package-refactor/release-prep work landed on
`master`.

## Deliverables for downstream work

- `TBLS.graph_strategy` / `TBLS.if_strategy` constructor parameters other
  future plans (e.g. Cython kernels) must keep supporting both branches of.
- `experiments/hyperparams.py` (new): the single place default
  hyperparameters and grid axes for `BLS`/`TBLS` live, so a future plan
  wiring up CCA/GFCCA multi-view fusion (see Non-goals) has an obvious place
  to add `CCA_*`/`GFCCA_*` grid axes next to the ones this plan adds.

## Current evidence and assumptions

### TBLS's current graph/IFS code path (`src/tbls/tbls.py::TBLS.fit`)

```python
k_mat = None
if self.use_if_weights or (self.graph_gamma > 0 and self.use_kernel_for_graph):
    k_mat = _kernel.compute_kernel_matrix(x_scaled)

if self.use_if_weights:
    s_mat = _ifs.compute_if_scores_geib(x_scaled, y_enc, K=k_mat, if_sigma=self.if_sigma)
else:
    s_mat = None

if self.graph_gamma > 0:
    l_mat = _graph.build_graph_laplacian(
        x_scaled, y_enc, K=k_mat,
        graph_alpha_in=self.graph_alpha_in, graph_alpha_p=self.graph_alpha_p,
        graph_knn=self.graph_knn, use_kernel=self.use_kernel_for_graph,
    )
else:
    l_mat = None
```

Both `use_if_weights` and `graph_gamma > 0` remain **opt-in** (default
`False`/`0.0` — this plan does not change whether these features are on by
default, only which formula is used *when* they're turned on).

### `GraphFuzzyKCCA`'s tuned formulas to port

- `_build_discriminative_graph(y)` (in `src/tbls/gfcca.py`): builds
  same-class/different-class adjacency `Ww`/`Wb` from labels only (no kNN, no
  distances), Laplacians `Lw = Dw - Ww`, `Lb = Db - Wb`, each normalized by
  `D^{-1/2} L D^{-1/2}` where `D = diag(abs(L).sum(axis=1) + 1e-8)` — **not**
  the same normalization convention as `_graph.build_graph_laplacian`'s
  `L = I - D^{-1/2} W D^{-1/2}`; port the formula exactly as GFCCA has it, do
  not "fix" it to match the other convention. Combined:
  `L = Lw_normalized - discriminative_beta * Lb_normalized`.
- `_ifs.compute_if_scores_simple(A, y, sigma_if, delta_if, min_weight)`
  already exists (extracted from `GraphFuzzyKCCA` during the earlier
  refactor) and returns a weight **vector** `s` of shape `(n,)` — TBLS needs
  `np.diag(s)` to match its own `S` (diagonal matrix) convention.

### Translating the user's constant block to actual parameter names

| User's constant | Current API name | Note |
|---|---|---|
| `TBLS_N_MAP_NODES` | `TBLS(n_map_trees=...)` | terminology drift ("nodes" vs "trees"), same value |
| `TBLS_N_ENHANCE_NODES` | `TBLS(n_enhance_trees=...)` | same |
| `TBLS_REG_PARAM` | `TBLS(reg_param=...)` | same |
| `TBLS_TREE_PARAMS["max_depth"]` | `TBLS(tree_max_depth=...)` | same |
| `TBLS_TREE_PARAMS["min_samples_split"]` | `TBLS(tree_min_samples_split=...)` | same |
| `TBLS_TREE_PARAMS["n_features_ratio"]` | `TBLS(tree_max_features_ratio=...)` | same value (0.7) |
| `TBLS_TREE_PARAMS["bootstrap_ratio"] = 0.632` | *(no such parameter)* | **not a gap** — 0.632 ≈ 1 − 1/e is simply the expected fraction of unique samples a Poisson(1) bootstrap includes; `RegressionTreeModule` already always does Poisson(1) bootstrap. Do not add a fake `bootstrap_ratio` parameter. |
| `TBLS_INCREMENTAL_METHOD = "spi"` | *(no method selector)* | `TBLS`'s only incremental-layer algorithm ("recompute weights", per its docstring) is the closest analog to `"spi"`. The legacy `main.py`'s other choices (`mse`, `pi`, `ge_if`) were never ported to the new `TBLS` and are **out of scope** for this plan — re-implementing them is new estimator functionality, not a config-wiring task. Flag, don't silently implement. |
| `BLS_*` (7 constants) | `BroadLearningSystem(n_feature_groups=..., n_feature_nodes_per_group=..., n_enhancement_groups=..., n_enhancement_nodes_per_group=..., map_func=..., enhance_func=..., reg_param=...)` | all match 1:1, no translation needed |
| `CCA_K`, `CCA_LAMBDA`, `KERNEL_GAMMA`, `GFCCA_GRAPH_GAMMA`, `GFCCA_SIGMA_GRAPH`, `DISCRIMINATIVE_BETA`, `GFCCA_SIGMA_IF`, `GFCCA_DELTA_IF` | `PairwiseKCCA`/`GraphFuzzyKCCA` constructor params | see Non-goals — not wired into `experiments/train.py` in this plan |

`GFCCA_SIGMA_GRAPH` is explicitly noted by the user as dead
("此参数已无用") — do not add it as a real parameter anywhere; if centralizing
constants for reference, keep it as a commented-out/annotated dead entry, not
a live config key.

### `experiments/train.py`'s current model wiring

`_cross_validate` (in `experiments/train.py`) always builds a `TBLS`, reading
only `model_cfg.get("map_num"|"enhance_num"|"reg_param")`.
`experiments/configs/default.yaml` already has an unused `model.name: tbls`
field. `BroadLearningSystem` is never instantiated by `train.py` today.

## Non-goals

- **CCA/GFCCA multi-view feature fusion in `experiments/train.py` is out of
  scope for this plan.** `experiments/dataprocess.py::DataLoader` and
  `experiments/train.py::_load_subsets` only handle a single feature matrix
  `X` per dataset key; `build_cca_features`/`build_gfcca_features` need a
  `list[X_views]` of *aligned* views of the same samples, and nothing in the
  current pipeline defines what the two (or more) views of
  `biomedical_larger.pkl`/`data_cross_train.pkl` would be. `experiments/classifiers.py`
  already has several multi-view-consuming comparison classifiers
  (`X_views` in MOFA/DIABLO/MOGONET/SNF/block PLS-DA) but no loader in the
  current pipeline actually produces `X_views` from these pkl files either —
  wiring CCA/GFCCA grid search requires first answering **"what are the two
  views in this dataset?"**, which only the user/their collaborator can
  answer. This plan centralizes `CCA_K`/`CCA_LAMBDA`/`KERNEL_GAMMA`/
  `GFCCA_*` as commented reference constants in `experiments/hyperparams.py`
  for a follow-up plan to activate, but does not wire them into any active
  code path.
- Re-implementing the legacy `mse`/`pi`/`ge_if` incremental-layer methods.
- Adding a `bootstrap_ratio` parameter to `RegressionTreeModule` (see
  translation table above — not a real gap).
- Cython kernels (unrelated, already tracked separately).
- Changing whether `use_if_weights`/`graph_gamma` are enabled by default —
  only which formula is used once they are enabled.

## Data model and storage changes

None (no new persisted schema). Grid-search output is additional Excel
sheets via the existing `TBLSResultSaver`, written to the same git-ignored
`results_dir/` — see Implementation step 4.

## Retrieval and tool contract changes

Not applicable (no retrieval/tool-calling system in this project).

## Workflow / prompt / agent changes

Not applicable.

## Implementation steps

### Step 1 — `src/tbls/_graph.py`: port the discriminative graph Laplacian

Add:

```python
def build_discriminative_graph_laplacian(
    y: NDArray[np.int64],
    discriminative_beta: float = 0.3,
) -> NDArray[np.float64]:
    """Label-only discriminative graph Laplacian ``L = Lw - beta * Lb``.

    Ported unchanged from :meth:`tbls.gfcca.GraphFuzzyKCCA._build_discriminative_graph`:
    a fully-connected (no kNN) same-class/different-class adjacency, each
    Laplacian symmetrically normalized by its own L1-degree (``D = diag(abs(L)
    .sum(axis=1) + 1e-8)``) -- a different normalization convention than
    :func:`build_graph_laplacian`'s ``I - D^{-1/2} W D^{-1/2}``; kept exactly
    as GFCCA has it, not reconciled with the other convention.

    Args:
        y: Integer class labels of shape ``(n,)``.
        discriminative_beta: Between-class penalty weight.

    Returns:
        Combined Laplacian of shape ``(n, n)``.
    """
```

Body: copy `GraphFuzzyKCCA._build_discriminative_graph`'s adjacency/Laplacian
construction plus its inline `normalize()` closure, then
`return lw_normalized - discriminative_beta * lb_normalized`. Do **not**
route this through `GraphFuzzyKCCA` (avoid a `tbls.tbls` → `tbls.gfcca`
import edge that would fight with `tbls.gfcca`'s own `from . import _ifs`);
both `gfcca.py` and `tbls.py` should call this shared `_graph` function
in a follow-up refactor of `gfcca.py` itself is optional and NOT required by
this plan (`gfcca.py`'s existing inline copy stays; duplication between the
two is acceptable here since `gfcca.py` already works and is out of scope —
see "Why" above for why we're not touching it).

Add a direct unit test in `tests/test_shared_modules.py`:
`test_build_discriminative_graph_laplacian_matches_gfcca_reference` —
reimplement the same formula independently (same style as the existing
`test_build_graph_laplacian_bandwidth_uses_full_distance_matrix`) and assert
bit-for-bit agreement, plus a symmetry/shape check. This is the same
"don't trust a shape-only test" lesson from the earlier accepted plan.

### Step 2 — `src/tbls/tbls.py`: strategy-switch parameters

Add five new `TBLS.__init__` parameters (stored as identically-named
attributes, per the estimator contract in `docs/architecture.md` section 5):

```python
graph_strategy: Literal["discriminative", "knn"] = "discriminative",
if_strategy: Literal["simple", "geib"] = "simple",
discriminative_beta: float = 0.3,
if_delta: float = 0.5,
if_min_weight: float = 1e-4,
```

Update `fit()`:

```python
need_kernel = (self.use_if_weights and self.if_strategy == "geib") or (
    self.graph_gamma > 0 and self.graph_strategy == "knn" and self.use_kernel_for_graph
)
k_mat = _kernel.compute_kernel_matrix(x_scaled) if need_kernel else None

if self.use_if_weights:
    if self.if_strategy == "simple":
        s_vec = _ifs.compute_if_scores_simple(
            x_scaled, y_enc,
            sigma_if=self.if_sigma, delta_if=self.if_delta, min_weight=self.if_min_weight,
        )
        s_mat = np.diag(s_vec)
    elif self.if_strategy == "geib":
        s_mat = _ifs.compute_if_scores_geib(x_scaled, y_enc, K=k_mat, if_sigma=self.if_sigma)
    else:
        raise ValueError(f"Unsupported if_strategy: {self.if_strategy!r}. Expected 'simple' or 'geib'.")
else:
    s_mat = None

if self.graph_gamma > 0:
    if self.graph_strategy == "discriminative":
        l_mat = _graph.build_discriminative_graph_laplacian(y_enc, discriminative_beta=self.discriminative_beta)
    elif self.graph_strategy == "knn":
        l_mat = _graph.build_graph_laplacian(
            x_scaled, y_enc, K=k_mat,
            graph_alpha_in=self.graph_alpha_in, graph_alpha_p=self.graph_alpha_p,
            graph_knn=self.graph_knn, use_kernel=self.use_kernel_for_graph,
        )
    else:
        raise ValueError(f"Unsupported graph_strategy: {self.graph_strategy!r}. Expected 'discriminative' or 'knn'.")
else:
    l_mat = None
```

Update the class docstring's `Args:` block with the five new parameters,
explicitly noting the default is the tuned GFCCA-derived formula and `"knn"`/`"geib"`
reproduce the estimator's original (pre-this-plan) behavior unchanged.

### Step 3 — Tests for the strategy switch

In `tests/test_tbls.py`:

- `test_tbls_discriminative_graph_and_simple_ifs_default` — fit with
  `use_if_weights=True, graph_gamma=0.1` and **no** explicit
  `graph_strategy`/`if_strategy` (i.e. exercises the new defaults); assert
  finite, non-degenerate predictions (same bar as the existing
  `test_tbls_ifs_and_graph_paths`).
- `test_tbls_knn_graph_and_geib_ifs_backward_compat` — fit with
  `use_if_weights=True, graph_gamma=0.1, graph_strategy="knn", if_strategy="geib"`;
  assert finite/non-degenerate. This is the regression guard that the old
  path remains fully reachable and functional.
- `test_tbls_invalid_strategy_raises` — `graph_strategy="bogus"` (with
  `graph_gamma>0`) and `if_strategy="bogus"` (with `use_if_weights=True`)
  each raise `ValueError` on `fit`.
- Rename the existing `test_tbls_ifs_and_graph_paths` docstring/comment to
  clarify it now exercises the *default* (discriminative/simple) strategy,
  or fold it into `test_tbls_discriminative_graph_and_simple_ifs_default`
  above rather than keeping a near-duplicate.

### Step 4 — `experiments/hyperparams.py` (new): centralized tunable defaults

```python
"""Centralized, directly-editable default hyperparameters and grid-search
axes for the experiments/ training pipeline.

Edit the *_DEFAULTS dicts to change the single-run default configuration;
edit the *_GRID dicts to change what --grid sweeps. Both are plain Python
dicts, not a YAML/CLI surface, by design -- this mirrors how the values were
originally tuned (as module-level constants), matching the request to keep
this directly editable rather than routed through another layer of config.
"""

BLS_DEFAULTS: dict = {
    "n_feature_groups": 30,
    "n_feature_nodes_per_group": 40,
    "n_enhancement_groups": 1,
    "n_enhancement_nodes_per_group": 500,
    "map_func": "relu",
    "enhance_func": "relu",
    "reg_param": 1.0,
}
BLS_GRID: dict = {
    "n_feature_groups": [15, 30, 60],
    "n_feature_nodes_per_group": [20, 40, 80],
    "reg_param": [0.1, 1.0, 10.0],
}

TBLS_DEFAULTS: dict = {
    "n_map_trees": 10,
    "n_enhance_trees": 10,
    "tree_max_depth": 5,
    "tree_min_samples_split": 3,
    "tree_max_features_ratio": 0.7,
    "reg_param": 1e-8,
    # graph_strategy/if_strategy intentionally left at TBLS's own defaults
    # ("discriminative"/"simple") rather than repeated here.
}
TBLS_GRID: dict = {
    "n_map_trees": [10, 20, 40],
    "n_enhance_trees": [10, 20, 40],
    "reg_param": [1e-8, 1e-4, 1e-2],
}

# Reserved for a future plan wiring up CCA/GFCCA multi-view feature fusion in
# this pipeline -- not read by any code path today. See docs/plan/
# 01-tbls-graph-ifs-strategy-and-grid-search.md, "Non-goals".
# CCA_K = 15
# CCA_LAMBDA = 0.1
# KERNEL_GAMMA = 1.0
# GFCCA_GRAPH_GAMMA = 0.5
# GFCCA_SIGMA_GRAPH = 0.5  # dead parameter, kept only for reference
# DISCRIMINATIVE_BETA = 0.3
# GFCCA_SIGMA_IF = 1.0
# GFCCA_DELTA_IF = 0.5
```

The exact grid values above are a starting point (small, roughly
order-of-magnitude neighbors of the user's tuned point estimates) — call this
out explicitly as editable/example in the module docstring and in
`docs/usage-experiments-cli.md`, not as a claim that this is the "correct"
search space.

### Step 5 — `experiments/train.py`: model selection + `--grid`

1. Add a `_build_model(model_cfg: dict) -> BaseEstimator` helper that reads
   `model_cfg.get("name", "tbls")` and returns either a `TBLS(**{**TBLS_DEFAULTS, **overrides})`
   or `BroadLearningSystem(**{**BLS_DEFAULTS, **overrides})`, where
   `overrides` come from whatever the YAML config / CLI already set
   (`map_num`/`enhance_num`/`reg_param` today — extend the config schema
   with model-appropriate keys, keeping backward compatibility with the
   existing `default.yaml` for the `tbls` case).
2. Add a `--grid` typer flag to the `train` command. When set:
   - Build `sklearn.model_selection.ParameterGrid(TBLS_GRID)` or
     `ParameterGrid(BLS_GRID)` depending on `model.name`.
   - For each grid point, run the existing `_cross_validate` k-fold loop
     with that configuration (merge grid point over `*_DEFAULTS`).
   - Collect each configuration's `TBLSEvaluator.calculate_average_metrics`
     output plus the configuration dict itself.
   - Write a `GridSummary` sheet via `TBLSResultSaver.save_summary` (already
     generic, not TBLS-specific despite the name — confirm this in code
     before assuming, adjust the call only if it turns out to hard-code
     TBLS-only column names) ranked by `avg_balanced_accuracy` descending,
     one row per grid point, alongside per-configuration per-fold sheets
     (reuse `save_fold_results` with a sheet name derived from the config,
     e.g. `Grid_{i:03d}`).
   - Log the winning configuration at the end (`logger.info`).
3. Without `--grid`, behavior is unchanged (single run with
   `*_DEFAULTS` merged with any config/CLI overrides — this **is** a
   behavior change from today's hardcoded-in-`_cross_validate` defaults, but
   the resulting default values must be verified identical to today's
   `experiments/configs/default.yaml` behavior for the `tbls` case, or called
   out explicitly if not).

### Step 6 — Tests for the CLI changes

- `tests/test_experiments_train.py` (new; note `experiments/` has no
  `__init__.py` today, matching the existing `pythonpath = ["."]` pytest
  config already used by `tests/test_real_dataset_smoke.py`'s
  `from experiments.smoke_run import ...`):
  - `_build_model({"name": "tbls"})` returns a `TBLS` instance with
    `TBLS_DEFAULTS` applied; same for `{"name": "bls"}` → `BroadLearningSystem`.
  - A `--grid` smoke test: on synthetic data (`sklearn.datasets.make_classification`,
    small `n_samples`), run the grid-search path with a tiny 2×2 grid
    (monkeypatch `TBLS_GRID`/`BLS_GRID` to 2 values each, not the full grid,
    so the test runs in milliseconds) and assert the resulting `GridSummary`
    has one row per grid point and is sorted by the ranking metric.

### Step 7 — Docs

- `docs/usage-tbls.md`: new "Graph and IFS strategy" section documenting
  `graph_strategy`/`if_strategy`/`discriminative_beta`/`if_delta`/`if_min_weight`,
  explicitly stating the default reproduces `GraphFuzzyKCCA`'s tuned formulas
  and `"knn"`/`"geib"` reproduce the pre-existing behavior.
- `docs/usage-experiments-cli.md`: new "Hyperparameter defaults and grid
  search" section covering `experiments/hyperparams.py`, `model.name: tbls|bls`,
  and `--grid` usage/output.
- `docs/architecture.md`: one-line addition to the shared-modules table
  noting `_graph.build_discriminative_graph_laplacian` and its relationship
  to `gfcca.py`'s inline copy (intentionally not deduplicated further — see
  Non-goals).

## Verification commands and test cases

```bash
uv run pytest tests/ -v            # includes new strategy + grid tests
uv run ruff check .
uv run ruff format --check .
uv run mypy src/tbls
uv run --group experiments python experiments/train.py --n-splits 2                 # unchanged default behavior, tbls
uv run --group experiments python experiments/train.py --n-splits 2 --grid          # new grid-search path
uv build && uvx twine check dist/*
```

Manually confirm (paste output into the acceptance report):

- `experiments/train.py --grid` run against a real sub-dataset (e.g. `DM` in
  `biomedical_larger.pkl`) actually produces a `GridSummary` sheet with
  distinguishable (not identical) rows per grid point.
- A `TBLS(use_if_weights=True, graph_gamma=0.1)` fit (new default) and a
  `TBLS(use_if_weights=True, graph_gamma=0.1, graph_strategy="knn",
  if_strategy="geib")` fit on the same real data produce **different**
  `predict_proba` outputs — sanity that the two strategies are not
  accidentally identical.

## Acceptance checklist

- [ ] `TBLS` has `graph_strategy`/`if_strategy`/`discriminative_beta`/
      `if_delta`/`if_min_weight` params; `get_params()`/`clone()` pick them up
      automatically (no manual `get_params` override needed).
- [ ] Default behavior (`graph_strategy="discriminative"`,
      `if_strategy="simple"`) is verified to reproduce `GraphFuzzyKCCA`'s
      formulas bit-for-bit via the new `_graph.py` unit test.
- [ ] `graph_strategy="knn"`/`if_strategy="geib"` reproduce the estimator's
      pre-this-plan behavior exactly (existing `test_build_graph_laplacian_*`
      tests in `test_shared_modules.py` still pass unmodified — they test
      `_graph.build_graph_laplacian` directly, independent of TBLS's default).
- [ ] Invalid strategy strings raise `ValueError` on `fit`, not a silent
      fallback or an `AttributeError` deep in `_graph`/`_ifs`.
- [ ] `experiments/hyperparams.py` exists with `BLS_DEFAULTS`/`BLS_GRID`/
      `TBLS_DEFAULTS`/`TBLS_GRID`; `CCA_*`/`GFCCA_*` are present only as
      commented reference constants, not wired into any active code path.
- [ ] `experiments/train.py` can train `BroadLearningSystem` via
      `model.name: bls` in the config, in addition to `tbls`.
- [ ] `experiments/train.py --grid` runs the full grid, writes a ranked
      `GridSummary` sheet, and logs the winning configuration.
- [ ] Full verification command table above passes, with real captured
      output (including the two "manually confirm" real-data checks).
- [ ] Docs updated per Step 7; no stale references to old defaults.

## Suggested commits

1. `feat(tbls): port GraphFuzzyKCCA's discriminative graph Laplacian into _graph.py`
2. `feat(tbls): add graph_strategy/if_strategy switch to TBLS (default: tuned GFCCA formulas)`
3. `test(tbls): strategy-switch coverage (default, backward-compat, invalid input)`
4. `feat(experiments): centralize BLS/TBLS hyperparameter defaults and grid axes`
5. `feat(experiments): model selection (tbls/bls) and --grid search in train.py`
6. `test(experiments): model-selection and grid-search smoke tests`
7. `docs: document the strategy switch and grid-search workflow`
