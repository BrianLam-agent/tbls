English | [简体中文](./usage-tbls.zh-CN.md)

# Using `TBLS`

`tbls.TBLS` is a Tree-based Broad Learning System classifier: an ensemble of
small regression trees arranged in the Broad Learning System's two-stage
"mapping → enhancement" architecture, trained with a closed-form
(pseudo-inverse) ridge solve instead of gradient descent, with optional
Intuitionistic Fuzzy Set (IFS) sample weighting and graph-Laplacian
regularization.

## Installation

```bash
pip install tbls
```

## Quickstart

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from tbls import TBLS

X, y = make_classification(n_samples=300, n_features=20, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = TBLS(n_map_trees=20, n_enhance_trees=20, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # shape (n_samples, n_classes)
```

`TBLS` works for binary and multi-class problems; `y` may be any label type
supported by `sklearn.preprocessing.LabelEncoder` (it is label-encoded
internally, and `model.classes_` holds the original labels in `predict`'s
output).

## Constructor parameters

| Parameter | Default | Meaning |
|---|---|---|
| `n_map_trees` | `20` | Number of mapping-stage regression trees. Each tree is trained on a Poisson(1) bootstrap of the input with Random Subspace feature sampling, and its scalar prediction becomes one mapped feature. |
| `n_enhance_trees` | `20` | Number of enhancement-stage trees, trained on the *mapped* features (not the raw input). |
| `n_increment_layers` | `0` | Number of additional enhancement layers appended after the initial fit (each recomputes the ridge solve with the enlarged feature matrix). |
| `tree_max_depth` | `5` | Max depth of every mapping/enhancement tree. |
| `tree_min_samples_split` | `3` | Min samples to split a node in every tree. |
| `tree_max_features_ratio` | `0.7` | Fraction of input features each tree samples (Random Subspace Method). |
| `reg_param` | `1e-4` | Ridge regularization strength for the output-weight solve. |
| `use_if_weights` | `False` | If `True`, weight training samples by their Intuitionistic Fuzzy Set credibility score (down-weights ambiguous/boundary samples). |
| `if_sigma` | `1.0` | Neighborhood-radius scale for the IFS score. |
| `graph_gamma` | `0.0` | Weight of graph-Laplacian regularization; `0` disables it entirely (no graph is even built). |
| `graph_alpha_in` / `graph_alpha_p` | `1.0` / `1.0` | Relative weight of the intrinsic (within-class) vs. penalty (between-class) graph Laplacian. |
| `graph_knn` | `10` | Number of nearest neighbors used to build the similarity graph. `<= 0` means fully connected. |
| `use_kernel_for_graph` | `True` | If `True`, graph distances are computed in RBF-kernel space rather than raw Euclidean space. |
| `random_state` | `None` | Seed for bootstrap sampling, feature subspace selection, and tree seeding. |
| `graph_strategy` | `"discriminative"` | Graph-Laplacian formula: `"discriminative"` (default, `GraphFuzzyKCCA`'s tuned label-only `Lw - beta*Lb`) or `"knn"` (the original kNN-graph). |
| `if_strategy` | `"simple"` | IFS scoring formula: `"simple"` (default, `GraphFuzzyKCCA`'s tuned per-class-center + relative-neighborhood) or `"geib"` (the original GEIB formulation). |
| `discriminative_beta` | `0.3` | Between-class penalty weight for `graph_strategy="discriminative"`. |
| `if_delta` | `0.5` | Relative neighborhood threshold for `if_strategy="simple"`. |
| `if_min_weight` | `1e-4` | Minimum IFS weight clip for `if_strategy="simple"`. |

`graph_threshold` and `class_sensitive` are reserved constructor parameters
kept for `sklearn.base.clone()`/`get_params()` compatibility but are not
currently used inside `fit`.

## When to turn on IFS weighting and graph regularization

Both are opt-in and add computational cost (an `O(n²)` kernel/distance matrix
and, for IFS, an `O(n·k)` neighborhood loop):

```python
model = TBLS(
    n_map_trees=20,
    n_enhance_trees=20,
    use_if_weights=True,      # down-weight ambiguous/boundary samples
    graph_gamma=0.1,          # enable graph-Laplacian regularization
    graph_knn=10,
    random_state=0,
)
model.fit(X_train, y_train)
```

Use them when your classes overlap heavily in feature space (IFS weighting)
or when you expect useful local/global class structure a plain ridge solve
would ignore (graph regularization). For clean, well-separated synthetic data
they usually make little difference — the sklearn-compatibility test suite
uses them primarily as a numerical-fidelity regression check (see
[`architecture.md`](./architecture.md#4-package-internals-shared-modules)),
not because they are always beneficial.

## Graph and IFS strategy

`TBLS` ships two graph-Laplacian formulas and two IFS-scoring formulas,
selectable independently. The defaults reproduce `GraphFuzzyKCCA`'s tuned
formulas (the same math, already verified inside `tbls.gfcca`); the
alternatives reproduce `TBLS`'s original pre-strategy-switch behavior exactly.

| Strategy | default | alternative |
|---|---|---|
| `graph_strategy` | `"discriminative"` -- label-only discriminative graph `L = Lw - beta*Lb` (no kNN, no distances), ported from `GraphFuzzyKCCA`. | `"knn"` -- the original kNN-graph `L = alpha_in*L_in - alpha_p*L_p` (`_graph.build_graph_laplacian`). |
| `if_strategy` | `"simple"` -- per-class Euclidean center distance + relative-neighborhood IFS (`_ifs.compute_if_scores_simple`). | `"geib"` -- the original GEIB formulation in kernel space (`_ifs.compute_if_scores_geib`). |

```python
# Default (tuned GFCCA formulas):
TBLS(use_if_weights=True, graph_gamma=0.1)
# Original TBLS behavior:
TBLS(use_if_weights=True, graph_gamma=0.1, graph_strategy="knn", if_strategy="geib")
```

`discriminative_beta`, `if_delta`, and `if_min_weight` parameterize the default
(`"discriminative"`/`"simple"`) formulas; they are ignored when the
corresponding strategy is set to `"knn"`/`"geib"`. An unsupported strategy
string raises `ValueError` on `fit`.

## Ablation variants (GTBLS/FTBLS/GFTBLS)

The graph term (`graph_gamma`, the "G" axis) and the fuzzy-IFS term
(`use_if_weights`, the "F" axis) are two independent switches, so all four
ablation combinations are reachable directly on `TBLS`:

| Variant | `use_if_weights` | `graph_gamma` | Meaning |
|---|---|---|---|
| `tbls` | `False` | `0.0` | Neither -- plain tree BLS. |
| `gtbls` | `False` | `> 0` | Graph regularization only. |
| `ftbls` | `True` | `0.0` | Fuzzy-IFS sample weighting only. |
| `gftbls` | `True` | `> 0` | Both (today's tuned default combination). |

For ablation studies, `build_tbls_variant` is a thin convenience factory that
picks the `(use_if_weights, graph_gamma)` pair for a name rather than adding a
third, potentially-conflicting constructor parameter (which would create two
ways to express the same state and complicate `clone()`/`get_params()`
round-tripping):

```python
from tbls import build_tbls_variant

gtbls = build_tbls_variant("gtbls", graph_gamma=0.2, n_map_trees=15)
ftbls = build_tbls_variant("ftbls", n_map_trees=15)
gftbls = build_tbls_variant("gftbls", graph_gamma=0.2, n_map_trees=15)
plain = build_tbls_variant("tbls", n_map_trees=15)
```

The factory forwards any other `TBLS` constructor keyword (`n_map_trees`,
`graph_strategy`, `if_strategy`, `random_state`, ...) unchanged. It raises
`ValueError` on an unknown variant, on `graph_gamma <= 0` for a graph-enabled
variant (`"gtbls"`/`"gftbls"` -- a non-positive value would silently disable
the graph term and defeat the ablation), and if `use_if_weights` is passed
explicitly (the factory's whole purpose is to set that unambiguously from
`variant`).

## Incremental layers

```python
model = TBLS(n_map_trees=20, n_enhance_trees=20, n_increment_layers=2, random_state=0)
model.fit(X_train, y_train)  # fits the base model, then appends 2 more enhancement layers
```

Each incremental layer adds `n_enhance_trees` more enhancement trees (trained
on the *current*, already-widened feature matrix) and recomputes the ridge
solve over the enlarged feature matrix. This trades training time for
capacity without discarding the already-fitted mapping trees.

## Using `TBLS` with scikit-learn tooling

Because `TBLS` is a standard `BaseEstimator`/`ClassifierMixin`, it works with
`cross_val_score`, `GridSearchCV`, `Pipeline`, etc.:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

scores = cross_val_score(TBLS(random_state=0), X, y, cv=5)

grid = GridSearchCV(
    TBLS(random_state=0),
    param_grid={"n_map_trees": [10, 20, 40], "reg_param": [1e-4, 1e-2]},
    cv=3,
)
grid.fit(X, y)
```

## Reproducibility

`random_state` controls every source of randomness inside `fit` (bootstrap
resampling, Random Subspace feature selection, per-tree seeds). Two `TBLS`
instances with the same `random_state` and the same input produce identical
output.

## Performance notes

- Training cost scales with `n_map_trees + n_enhance_trees * (1 +
  n_increment_layers)` regression-tree fits, plus one `O(n²)` kernel/distance
  computation if `use_if_weights` or `graph_gamma > 0`.
- For a first pass on a new dataset, start small (`n_map_trees=10,
  n_enhance_trees=10`, `use_if_weights=False`, `graph_gamma=0.0`) — this is
  exactly what `experiments/smoke_run.py` does to sanity-check a dataset in
  seconds; see [`usage-experiments-cli.md`](./usage-experiments-cli.md).
- The RBF kernel and graph-Laplacian computations are `O(n²)` in the number
  of *training* samples; for very large datasets, subsample before enabling
  `use_if_weights`/`graph_gamma`.
