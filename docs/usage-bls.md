English | [简体中文](./usage-bls.zh-CN.md)

# Using `BroadLearningSystem`

`tbls.BroadLearningSystem` is a classic Broad Learning System classifier:
random-weight feature-mapping nodes, random-weight enhancement nodes, and a
ridge pseudo-inverse output layer, with optional class weighting and true
incremental-enhancement learning via the Woodbury matrix identity (no
retraining from scratch).

## Quickstart

```python
from sklearn.datasets import make_classification
from tbls import BroadLearningSystem

X, y = make_classification(n_samples=300, n_features=20, random_state=0)

model = BroadLearningSystem(
    n_feature_groups=10,
    n_feature_nodes_per_group=100,
    n_enhancement_groups=10,
    n_enhancement_nodes_per_group=100,
    random_state=0,
)
model.fit(X, y)
model.predict(X[:5])
model.predict_proba(X[:5])
```

## Constructor parameters

| Parameter | Default | Meaning |
|---|---|---|
| `n_feature_groups` | `10` | Number of independent feature-mapping node groups. |
| `n_feature_nodes_per_group` | `100` | Nodes per feature-mapping group. |
| `n_enhancement_groups` | `10` | Number of enhancement-node groups (built on top of the mapped features). |
| `n_enhancement_nodes_per_group` | `100` | Nodes per enhancement group. |
| `map_func` | `"relu"` | Activation for feature-mapping nodes: `relu`, `sigmoid`, `tanh`, `linear`, or `leaky_relu`. |
| `enhance_func` | `"relu"` | Activation for enhancement nodes (same choices). |
| `reg_param` | `1e-8` | Ridge regularization parameter for the pseudo-inverse solve. |
| `class_weights` | `None` | `None` (unweighted), `"auto"` (inverse class-frequency weighting via `sklearn.utils.class_weight.compute_class_weight`), or an explicit `{class_label: weight}` dict. |
| `random_state` | `None` | Seed for the random feature/enhancement weight matrices. |

## Class imbalance

```python
model = BroadLearningSystem(class_weights="auto", random_state=0)
# or, with explicit weights:
model = BroadLearningSystem(class_weights={0: 1.0, 1: 5.0}, random_state=0)
```

## Incremental enhancement (Woodbury update)

Unlike `TBLS`'s incremental layers (which recompute the full ridge solve),
`BroadLearningSystem.incremental_enhance` updates the pseudo-inverse
in closed form without re-solving from scratch:

```python
model = BroadLearningSystem(random_state=0)
model.fit(X_train, y_train)

# Later, add capacity without discarding the fitted output weights:
model.incremental_enhance(X_train, num_new_nodes=100)
model.predict(X_test)
```

`X` passed to `incremental_enhance` must be the *same* training data used in
`fit` (or a consistent replacement batch) — it recomputes the existing
feature/enhancement nodes for that data to derive the correction term; it is
not a mechanism for training on new samples.

## Missing values

`fit`/`predict`/`predict_proba` all pass their input through
`np.nan_to_num(..., nan=0.0)` before scaling — `NaN` values are silently
replaced with `0.0` after standardization is undone conceptually (i.e. before
the scaler sees them). If your data has meaningful missing values, impute
them yourself (e.g. `sklearn.impute.SimpleImputer`) rather than relying on
this fallback.

## Using with scikit-learn tooling

`BroadLearningSystem` is a standard `BaseEstimator`/`ClassifierMixin` and
works with `cross_val_score`, `GridSearchCV`, `Pipeline`, etc., exactly like
`TBLS` — see [`usage-tbls.md`](./usage-tbls.md#using-tbls-with-scikit-learn-tooling).
