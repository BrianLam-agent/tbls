[English](./models.md) | 简体中文

# `model.name` — YAML 配置里写什么

YAML 的 `model.name` 字段决定训练哪个估计器。分两个层级:

1. **`tbls`** 和 **`bls`** 是包内的估计器。默认值取自
   `experiments/hyperparams.py`(下表列出),YAML `model:` 下的键可覆盖。
2. **任何其他名字**是比较*基线*,由 `experiments.classifiers.create_classifier`
   构建。它是**固定名单**(不是任意字符串 — `model.name: 沈博琛` **不合法**,
   会直接抛 `ValueError("Unknown classifier ...")`)。每个基线自带类别均衡
   默认(见每行),你再用额外 YAML `model:` 键覆盖底层 sklearn 构造器的任何
   kwarg。

一个完整的配置示例:用 `Logistic Regression` 对比 `TBLS`:

```yaml
dataset: biomedical_larger
run_name: My experiment
model: {name: lr, C: 2.0}
preprocess: {feature_selection: lasso, resampling: smote}
cv: {n_splits: 5, random_state: 42}
output_dir: examples/runs
```

## 第 1 层:包内估计器

### `tbls` — Tree-based Broad Learning System

完整 TBLS 构造器文档在 [`../usage-tbls.md`](../usage-tbls.md)。当你写
`model.name: tbls`,`experiments/train.py` 构建
`tbls.TBLS(**{defaults, **YAML_overrides})`,默认值:

| 参数 | 默认 |
|---|---|
| `n_map_trees` | `10` |
| `n_enhance_trees` | `10` |
| `tree_max_depth` | `5` |
| `tree_min_samples_split` | `3` |
| `tree_max_features_ratio` | `0.7` |
| `reg_param` | `1e-8` |
| `graph_strategy` | `"discriminative"` |
| `if_strategy` | `"simple"` |
| `use_if_weights` / `graph_gamma` / `if_sigma` / `if_delta` / `discriminative_beta` / `graph_knn` / 其他 | `TBLS` 构造器自身默认(见 `usage-tbls.md`) |

在 YAML 的 `model:` 下加任何上述参数即可覆盖(旧键 `map_num` → `n_map_trees`、
`enhance_num` → `n_enhance_trees` 仍接受,向后兼容)。

### `bls` — 经典 Broad Learning System

经 `tbls.BroadLearningSystem` 构建,默认值:

| 参数 | 默认 |
|---|---|
| `n_feature_groups` | `30` |
| `n_feature_nodes_per_group` | `40` |
| `n_enhancement_groups` | `1` |
| `n_enhancement_nodes_per_group` | `500` |
| `map_func` | `"relu"` |
| `enhance_func` | `"relu"` |
| `reg_param` | `1.0` |

完整构造器参考:[`../usage-bls.md`](../usage-bls.md)。

## 第 2 层:基线(来自 `experiments.classifiers.create_classifier`)

每行列出:YAML `model.name`、底层 sklearn 估计器、若你只写 `name` 不写其他
键时**继承的默认**、YAML `model:` 键如何转发(底层 sklearn 类接受的任何
kwarg)、是否需要额外运行时依赖。

"已自带类别均衡"列的意思:你不必非得开 `resampling: smote` 才能拿到类别
均衡的拟合 — 每个支持的基线要么有 `class_weight="balanced"`、要么是其估计器
特定等价物默认就开。(`knn`/`nb`/`lda`/`gbdt`/`mlp`/`dnn` 没有 class-weight
kwarg — 这些请配合 `resampling: smote`。)

| `model.name` | 包装 | 内置默认(YAML 可覆盖) | 类别均衡 | 额外依赖? |
|---|---|---|---|---|
| `rf` | `RandomForestClassifier` | `n_estimators=200`,`class_weight="balanced_subsample"`,`n_jobs=-1` | 是 | 否 |
| `svm` | `SVC` | `C=1.0`,`kernel="rbf"`,`gamma="scale"`,`class_weight="balanced"`,`probability=True` | 是 | 否 |
| `xgb` | `BalancedXGBClassifier`(自动类别加权) | — | 是 | **xgboost**(在 `experiments` 组内) |
| `lgb` | `BalancedLGBMClassifier` | — | 是 | **lightgbm**(不在组内 — 须单独 `pip install lightgbm`) |
| `catboost` | `BalancedCatBoostClassifier` | — | 是 | **catboost**(不在 — `pip install catboost`) |
| `knn` | `KNeighborsClassifier` | `n_neighbors=5`,`weights="distance"`,`n_jobs=-1` | 否 — 用 `resampling: smote` | 否 |
| `lr` | `LogisticRegression` | `C=1.0`,`class_weight="balanced"`,`solver="lbfgs"`,`max_iter=1000` | 是 | 否 |
| `lasso` | `LogisticRegression`(`penalty="l1"`,`solver="saga"`) | l1 正则 LR,`C=1.0`,`class_weight="balanced"`,`max_iter=1000` | 是 | 否 |
| `elasticnet` | `LogisticRegression`(`penalty="elasticnet"`,`solver="saga"`) | 弹性网 LR,`l1_ratio=0.5`,`C=1.0`,`class_weight="balanced"`,`max_iter=1000` | 是 | 否 |
| `nb` | `GaussianNB` | — | 否 — 用 `resampling: smote` | 否 |
| `lda` | `LinearDiscriminantAnalysis` | `solver="svd"` | 否 — 用 `resampling: smote` | 否 |
| `cart` | `DecisionTreeClassifier` | `class_weight="balanced"` | 是 | 否 |
| `mlp` | `MLPClassifier` | `hidden_layer_sizes=(100,)`,`activation="relu"`,`solver="adam"`,`max_iter=300` | 否 — 用 `resampling: smote` | 否 |
| `dnn` | `MLPClassifier`(更深层) | `hidden_layer_sizes=(100,100,100)`,`solver="adam"`,`learning_rate="adaptive"`,`learning_rate_init=0.001`,`max_iter=300`,`early_stopping=True` | 否 — 用 `resampling: smote` | 否 |
| `extratrees` | `ExtraTreesClassifier` | `n_estimators=200`,`class_weight="balanced_subsample"`,`n_jobs=-1` | 是 | 否 |
| `gbdt` | `GradientBoostingClassifier` | `n_estimators=100`,`learning_rate=0.1`,`max_depth=3` | 否 — 用 `resampling: smote` | 否 |

上表以外的字符串(加上多视图场景的 `block_plsda`/`block_splsda`/
`mogonet`/`mogonet_nn`/`mofa`/`diablo`/`snf`,这些单视图实验不用 — 若
确有需要见 `experiments/classifiers.py`)都会**以一个清晰的 ValueError 拒绝**,
并列出支持的名字 — 不会有打字错误的静默跑。

## 传额外的构造器参数

任何 `model.name`(两层都一样),YAML `model:` 下的键(除 `name` 本身外)都会
作为关键字参数转发给估计器构造器。例子 — 更宽的 `RandomForest`:

```yaml
model:
  name: rf
  n_estimators: 500
  max_depth: 10
  random_state: 7   # 任何模型都从 model.random_state 读(默认 42)
```

## `--grid` 与基线

`--grid` 开箱只扫包内估计器的网格(`TBLS_GRID`/`BLS_GRID`)。**基线没有默认
网格**:对 `model.name: lr` 传 `--grid` 会退化为单次 k-fold 运行并打告警
(不会去扫假网格);要真扫基线,就在 YAML 里显式写 `grid:` 段:

```yaml
model: {name: lr}
grid:
  C: [0.01, 0.1, 1.0, 10.0]
```

完整网格规则:[grid-search.md](grid-search.md)。
