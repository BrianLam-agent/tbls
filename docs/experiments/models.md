English | [简体中文](./models.zh-CN.md)

# `model.name` — what to put in a YAML config

The YAML `model.name` field selects what gets fit. There are two tiers:

1. **`tbls`** and **`bls`** are the package's own estimators. They take their
   defaults from `experiments/hyperparams.py` (shown below), overridable by the
   YAML `model:` keys.
2. **Anything else** is a comparison *baseline* built by
   `experiments.classifiers.create_classifier`. There is a fixed list (not
   arbitrary strings — `model.name: 沈博琛` is **not** valid and would
   literally raise `ValueError("Unknown classifier ...")`). Each baseline
   carries built-in class-balancing defaults (see per-model rows); you then
   override any underlying sklearn constructor kwarg with extra YAML `model:`
   keys.

A full config example that runs `Logistic Regression` against `TBLS`:

```yaml
dataset: biomedical_larger
run_name: My experiment
model: {name: lr, C: 2.0}
preprocess: {feature_selection: lasso, resampling: smote}
cv: {n_splits: 5, random_state: 42}
output_dir: examples/runs
```

## Tier 1: package estimators

### `tbls` — Tree-based Broad Learning System

The full TBLS constructor is documented in
[`../usage-tbls.md`](../usage-tbls.md). When you write `model.name: tbls`,
`experiments/train.py` builds `tbls.TBLS(**{defaults, **YAML_overrides})` with
the defaults:

| Parameter | Default |
|---|---|
| `n_map_trees` | `10` |
| `n_enhance_trees` | `10` |
| `tree_max_depth` | `5` |
| `tree_min_samples_split` | `3` |
| `tree_max_features_ratio` | `0.7` |
| `reg_param` | `1e-8` |
| `graph_strategy` | `"discriminative"` |
| `if_strategy` | `"simple"` |
| `use_if_weights` / `graph_gamma` / `if_sigma` / `if_delta` / `discriminative_beta` / `graph_knn` / etc. | the `TBLS` constructor's own defaults (see `usage-tbls.md`) |

Override any of these by adding it under `model:` in your YAML (legacy
`map_num` → `n_map_trees` and `enhance_num` → `n_enhance_trees` are still
accepted for backward compatibility).

### `bls` — classic Broad Learning System

Built via `tbls.BroadLearningSystem` with the defaults:

| Parameter | Default |
|---|---|
| `n_feature_groups` | `30` |
| `n_feature_nodes_per_group` | `40` |
| `n_enhancement_groups` | `1` |
| `n_enhancement_nodes_per_group` | `500` |
| `map_func` | `"relu"` |
| `enhance_func` | `"relu"` |
| `reg_param` | `1.0` |

Full constructor reference: [`../usage-bls.md`](../usage-bls.md).

## Tier 2: baselines (from `experiments.classifiers.create_classifier`)

Each row shows: the YAML `model.name`, the wrapped sklearn estimator, the
built-in defaults you inherit if you specify *only* `name` and nothing else,
which YAML `model:` keys are forwarded (any kwarg the underlying sklearn class
accepts), and whether an extra runtime dependency is required.

The "_class balancing already built in_" column means you don't have to set
`resampling: smote` to get a class-balanced fit — every supported baseline
either gets `class_weight="balanced"` or its estimator-specific equivalent by
default. (`knn`/`nb`/`lda`/`gbdt`/`mlp`/`dnn` have no class-weight kwarg — use
`resampling: smote` for them.)

| `model.name` | Wraps | Built-in defaults (YAML override possible) | Class balance | Extra dep? |
|---|---|---|---|---|
| `rf` | `RandomForestClassifier` | `n_estimators=200`, `class_weight="balanced_subsample"`, `n_jobs=-1` | yes | no |
| `svm` | `SVC` | `C=1.0`, `kernel="rbf"`, `gamma="scale"`, `class_weight="balanced"`, `probability=True` | yes | no |
| `xgb` | `BalancedXGBClassifier` (auto class weighting) | — | yes | **xgboost** (in `experiments` group) |
| `lgb` | `BalancedLGBMClassifier` | — | yes | **lightgbm** (NOT in the group — must `pip install lightgbm` separately) |
| `catboost` | `BalancedCatBoostClassifier` | — | yes | **catboost** (NOT — `pip install catboost`) |
| `knn` | `KNeighborsClassifier` | `n_neighbors=5`, `weights="distance"`, `n_jobs=-1` | no — use `resampling: smote` | no |
| `lr` | `LogisticRegression` | `C=1.0`, `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=1000` | yes | no |
| `lasso` | `LogisticRegression` (`penalty="l1"`, `solver="saga"`) | l1-penalized LR, `C=1.0`, `class_weight="balanced"`, `max_iter=1000` | yes | no |
| `elasticnet` | `LogisticRegression` (`penalty="elasticnet"`, `solver="saga"`) | elastic-net LR, `l1_ratio=0.5`, `C=1.0`, `class_weight="balanced"`, `max_iter=1000` | yes | no |
| `nb` | `GaussianNB` | — | no — use `resampling: smote` | no |
| `lda` | `LinearDiscriminantAnalysis` | `solver="svd"` | no — use `resampling: smote` | no |
| `cart` | `DecisionTreeClassifier` | `class_weight="balanced"` | yes | no |
| `mlp` | `MLPClassifier` | `hidden_layer_sizes=(100,)`, `activation="relu"`, `solver="adam"`, `max_iter=300` | no — use `resampling: smote` | no |
| `dnn` | `MLPClassifier` (deeper) | `hidden_layer_sizes=(100,100,100)`, `solver="adam"`, `learning_rate="adaptive"`, `learning_rate_init=0.001`, `max_iter=300`, `early_stopping=True` | no — use `resampling: smote` | no |
| `extratrees` | `ExtraTreesClassifier` | `n_estimators=200`, `class_weight="balanced_subsample"`, `n_jobs=-1` | yes | no |
| `gbdt` | `GradientBoostingClassifier` | `n_estimators=100`, `learning_rate=0.1`, `max_depth=3` | no — use `resampling: smote` | no |

Anything other than the strings above (plus the multi-view corner cases
`block_plsda`/`block_splsda`/`mogonet`/`mogonet_nn`/`mofa`/`diablo`/`snf`, which
are out of scope for these single-view experiments — see `experiments/classifiers.py`
if for some reason you need them) is **rejected with a clear ValueError**
naming the supported options — no silent typo-run.

## Passing extra constructor parameters

For any `model.name` (both tiers), the YAML keys under `model:` (other than
`name` itself) are forwarded as keyword arguments to the estimator's
constructor. Example — wider `RandomForest`:

```yaml
model:
  name: rf
  n_estimators: 500
  max_depth: 10
  random_state: 7   # any model reads this from model.random_state (default 42)
```

## `--grid` and baselines

`--grid` only sweeps the package estimator grids (`TBLS_GRID`/`BLS_GRID`) out of
the box. **Baselines have no default grid**: passing `--grid` with
`model.name: lr` falls back to a single k-fold run and logs a warning; to
actually sweep a baseline, give it a YAML `grid:` section explicitly:

```yaml
model: {name: lr}
grid:
  C: [0.01, 0.1, 1.0, 10.0]
```

Full grid rules: [grid-search.md](grid-search.md).