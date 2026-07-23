English | [简体中文](./usage-cca-gfcca.zh-CN.md)

# Using `PairwiseKCCA` and `GraphFuzzyKCCA`

Both classes are **two-view** kernel Canonical Correlation Analysis (CCA)
feature extractors: given two feature matrices describing the *same* samples
from two different views (e.g. two modalities, two feature subsets), they
find projection directions in kernel space that maximize correlation between
the views, and can then project new samples from either view into that shared
space. See [`architecture.md`](./architecture.md#5-estimator-contract) for
why they intentionally do not implement sklearn's `TransformerMixin`.

- `PairwiseKCCA` — plain regularized kernel CCA.
- `GraphFuzzyKCCA` — adds Intuitionistic Fuzzy Set sample credibility scoring
  and a discriminative graph-embedding regularizer (uses class labels `y`
  during `fit`, unlike `PairwiseKCCA`).

## `PairwiseKCCA`

```python
import numpy as np
from tbls import PairwiseKCCA

X1_train, X2_train = np.random.randn(100, 30), np.random.randn(100, 20)
X1_test, X2_test = np.random.randn(20, 30), np.random.randn(20, 20)

cca = PairwiseKCCA(k=7, reg_lambda=0.1, kernel_gamma=0.1)
cca.fit(X1_train, X2_train)

Z1_train, Z2_train = cca.transform()          # training projections, both views
Z1_test = cca.transform_view1(X1_test)         # project new view-1 samples
Z2_test = cca.transform_view2(X2_test)         # project new view-2 samples
```

| Parameter | Default | Meaning |
|---|---|---|
| `k` | `7` | Number of canonical variable pairs kept. |
| `reg_lambda` | `0.1` | Ridge regularization added to each view's kernel matrix (numerical stability). |
| `kernel_gamma` | `0.1` | Base RBF kernel width passed to `tbls._kernel.rbf_kernel` (still adaptively scaled by the median pairwise distance). |

## `GraphFuzzyKCCA`

Same two-view shape, but `fit` additionally takes class labels `y` and builds
a discriminative graph (`Lw - beta * Lb`) plus IFS credibility weights:

```python
from tbls import GraphFuzzyKCCA

gfcca = GraphFuzzyKCCA(k=7, reg_lambda=0.1, kernel_gamma=0.1, graph_gamma=0.5)
gfcca.fit(X1_train, X2_train, y_train)

Z1_train, Z2_train = gfcca.transform()
Z1_test = gfcca.transform_view1(X1_test)
Z2_test = gfcca.transform_view2(X2_test)
```

Key parameters beyond `PairwiseKCCA`'s:

| Parameter | Default | Meaning |
|---|---|---|
| `graph_gamma` | `0.1` | Weight of the discriminative graph-embedding term. |
| `sigma_if` / `delta_if` | `1.0` / `0.5` | IFS membership width / relative neighborhood threshold. |
| `min_weight` | `1e-4` | Minimum IFS weight clip (prevents a singular weighted kernel matrix). |
| `discriminative_beta` | `0.3` | Between-class penalty weight in the discriminative graph. |
| `max_attempts` / `tau_factor` (passed to `fit`) | `5` / `10.0` | On a non-positive-definite generalized eigenproblem, `fit` retries up to `max_attempts` times, scaling up its numerical-stability term by `tau_factor` each time, before giving up and raising. |

## Multi-view feature-building pipelines

For more than two views, use the module-level helpers, which run every
pairwise combination of views and concatenate the results:

```python
from tbls.cca import build_cca_features, project_cca_features
# or: from tbls.gfcca import build_gfcca_features (needs y); project_cca_features
#     is intentionally not re-exported from gfcca — see architecture.md §4.1.

X_views_train = [X1_train, X2_train, X3_train]
F_train, cca_models = build_cca_features(X_views_train, cca_k=7)

X_views_test = [X1_test, X2_test, X3_test]
F_test = project_cca_features(X_views_test, cca_models)
```

`cca_models` maps each `(i, j)` view-pair index to its fitted `PairwiseKCCA`
(or `GraphFuzzyKCCA`, via `tbls.gfcca.build_gfcca_features`) instance, so
`project_cca_features` can reproduce the *training-time* projection exactly
for held-out data — there is no data leakage between train and test.

## No `Pipeline` support (by design)

Because both classes need two aligned feature matrices at `fit` time and
"which view" at `transform` time, they cannot honor
`sklearn.pipeline.Pipeline`'s single-`X` contract without silently dropping a
view. If you need one, write a small adapter, e.g.:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class SingleViewCCA(BaseEstimator, TransformerMixin):
    """Adapts PairwiseKCCA to a single stacked-view Pipeline step."""

    def __init__(self, n_features_view1: int, **cca_kwargs):
        self.n_features_view1 = n_features_view1
        self.cca_kwargs = cca_kwargs

    def fit(self, X, y=None):
        x1, x2 = X[:, : self.n_features_view1], X[:, self.n_features_view1 :]
        self.cca_ = PairwiseKCCA(**self.cca_kwargs).fit(x1, x2)
        return self

    def transform(self, X):
        x1, x2 = X[:, : self.n_features_view1], X[:, self.n_features_view1 :]
        z1 = self.cca_.transform_view1(x1)
        z2 = self.cca_.transform_view2(x2)
        return np.hstack([z1, z2])
```

This is not shipped in `tbls` itself because the "stack the two views into
one `X`" convention is an application-specific choice (column ranges, or a
tuple, or a `dict`), not something the library can decide generically.
