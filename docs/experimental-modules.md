English | [简体中文](./experimental-modules.zh-CN.md)

# `tbls.genoptim` and `tbls.ensemble` (experimental)

Both subpackages ship inside the `tbls` wheel (they need nothing beyond
numpy/scipy/scikit-learn) but are **experimental**: importing either emits a
`FutureWarning`, and their public API may change without notice between minor
versions, without following the deprecation policy the core estimators
(`TBLS`, `BroadLearningSystem`, `PairwiseKCCA`, `GraphFuzzyKCCA`) follow.

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)  # acknowledge experimental status
    from tbls.ensemble import TreeSelector, diversity_score
    from tbls.genoptim import ChromosomeEncoder, PopulationInitializer
```

## `tbls.ensemble` — fully functional

Two standalone pieces, no coupling to `TBLS` internals:

### `diversity_metrics`

```python
from tbls.ensemble import diversity_score, jaccard_similarity

feature_sets = [{1, 2, 3}, {2, 3, 4}, {5, 6}]
diversity_score(feature_sets, method="jaccard")   # mean pairwise (1 - Jaccard similarity)
diversity_score(feature_sets, method="entropy")   # Shannon entropy of feature-occurrence frequency
```

### `TreeSelector`

A generic top-k / threshold selector over a `{index: fitness_score}` dict —
does not know or care that the indices came from `TBLS` trees; works with any
fitness/diversity scoring you provide:

```python
from tbls.ensemble import TreeSelector

selector = TreeSelector(selection_method="top_k", weight_method="performance")
selector.fit(fitness_scores={0: 0.8, 1: 0.6, 2: 0.9, 3: 0.4})
selector.get_selected_trees()   # indices of the selected subset
selector.get_weights()          # normalized weights for the selected subset
```

## `tbls.genoptim` — partially functional

| Component | Status |
|---|---|
| `ChromosomeEncoder`, `PopulationInitializer` | Functional. Pure encode/decode and bootstrap-population utilities; no `TBLS` coupling. |
| `operators.selection` / `crossover` / `mutation` | Functional. Standard GA operators (tournament/roulette selection, uniform/single-point crossover, bit-flip/Gaussian/adaptive mutation) on plain arrays. |
| `fitness.MultiObjectiveFitness` | **Not verified against the current `TBLS`.** |
| `ga_optimizer.GeneticOptimizer` | **Not verified against the current `TBLS`.** |

### Why `fitness.py`/`ga_optimizer.py` don't work against `tbls.tbls.TBLS`

These two modules were written against an older, since-removed estimator
(`TreeBroadLearningSystem`) and call attributes that do not exist on the
current `tbls.tbls.TBLS`:

| Called in `genoptim` | Exists on legacy `TreeBroadLearningSystem` | Exists on current `tbls.tbls.TBLS`? |
|---|---|---|
| `model.predict(X, trees=selected_trees)` | yes | **no** — `predict(X)` has no `trees=` kwarg |
| `model.mapping_trees` | yes | **no** — the equivalent attribute is `map_trees_` |
| `tree.selected_features` | yes | **no** — the equivalent attribute is `RegressionTreeModule.feature_indices_` |
| `model.tree_params["bootstrap_ratio"]` | yes | **no** such attribute |
| `model.n_map_nodes` | yes | **no** — the equivalent constructor arg is `n_map_trees` |
| `model.X_original` | yes | **no** such attribute |

Calling `MultiObjectiveFitness.calculate(model, ...)` or
`GeneticOptimizer.optimize(model, ...)` with a `tbls.tbls.TBLS` instance will
raise `AttributeError`/`TypeError` at the first of these accesses. This is
**not a bug to silently patch** — it reflects a genuine capability gap: the
current `TBLS` has no notion of "predict using only a subset of trees" or
per-tree feature-index introspection in the shape `genoptim` expects.

### What it would take to fix this

Making `genoptim` work against the current `TBLS` requires deciding on and
implementing new `TBLS` capabilities, not just renaming attributes:

1. A `predict`/`predict_proba` variant that accepts a tree subset (or weights
   per tree) rather than always using the full mapping+enhancement ensemble.
2. Deciding whether tree selection operates on mapping trees, enhancement
   trees, or both, and how a selected subset's outputs are recombined for the
   output-weight solve (`TBLS` currently trains `W_` jointly over the full
   stacked feature matrix `A`, not per-tree).
3. Updating `fitness.py`/`ga_optimizer.py` to the resulting API and adding an
   end-to-end test that actually runs `GeneticOptimizer.optimize(...)` against
   a fitted `TBLS` and asserts on the result — until such a test exists, do
   not assume any change to these modules works.

This is intentionally out of scope for the current release — it's new
estimator functionality, not a packaging/refactor task.

## If you want to use genetic tree selection today

Use `tbls.ensemble`'s standalone pieces directly against your own
tree-subset/prediction logic (e.g. store per-tree predictions yourself and
combine them under a `ChromosomeEncoder`-decoded mask), rather than through
`genoptim.fitness`/`ga_optimizer`, until the gap above is closed.
