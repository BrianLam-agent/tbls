English | [简体中文](./models.zh-CN.md)

# `model.name` — YAML 里写什么

YAML 的 `model.name` 决定训练哪个模型，分两层：

1. **`tbls`** 和 **`bls`** 是包自带的估计器，默认值取自
   `experiments/hyperparams.py`（下表列出），YAML `model:` 下的键可覆盖。
2. **其他名字**是对比基线，由 `experiments.classifiers.create_classifier`
   构建。名字是**固定列表**——`model.name: 沈博琛` **不合法**，会直接抛
   `ValueError("Unknown classifier ...")`。每个基线自带类别均衡默认值
   （见下表），你可以用 YAML `model:` 下的额外键覆盖底层 sklearn 构造器
   的任何参数。

完整示例——用 Logistic Regression 对比 TBLS：

```yaml
dataset: biomedical_larger
run_name: My experiment
model: {name: lr, C: 2.0}
preprocess: {feature_selection: lasso, resampling: smote}
cv: {n_splits: 5, random_state: 42}
output_dir: examples/runs
```

## 第一层：包内估计器

### `tbls` — Tree-based Broad Learning System

完整构造器文档见 [`../usage-tbls.md`](../usage-tbls.md)。写
`model.name: tbls` 时，`train.py` 用以下默认值构建 `tbls.TBLS`，YAML
中给出的键会覆盖对应默认值：

| 参数 | 默认值 |
|---|---|
| `n_map_trees` | `10` |
| `n_enhance_trees` | `10` |
| `tree_max_depth` | `5` |
| `tree_min_samples_split` | `3` |
| `tree_max_features_ratio` | `0.7` |
| `reg_param` | `1e-8` |
| `graph_strategy` | `"discriminative"` |
| `if_strategy` | `"simple"` |
| `use_if_weights` / `graph_gamma` / `if_sigma` / `if_delta` / `discriminative_beta` / `graph_knn` / 其他 | `TBLS` 构造器自身默认值（见 `usage-tbls.md`） |

旧键 `map_num` → `n_map_trees`、`enhance_num` → `n_enhance_trees` 仍可
用，向后兼容。

### `bls` — 经典 Broad Learning System

通过 `tbls.BroadLearningSystem` 构建，默认值：

| 参数 | 默认值 |
|---|---|
| `n_feature_groups` | `30` |
| `n_feature_nodes_per_group` | `40` |
| `n_enhancement_groups` | `1` |
| `n_enhancement_nodes_per_group` | `500` |
| `map_func` | `"relu"` |
| `enhance_func` | `"relu"` |
| `reg_param` | `1.0` |

完整构造器参考：[`../usage-bls.md`](../usage-bls.md)。

## 第二层：基线（来自 `create_classifier`）

下表列出所有支持的基线名、底层 sklearn 估计器、只写 `name` 不写其他键时
**继承的默认值**、是否自带类别均衡、是否需要额外依赖。

"自带类别均衡"意味着你不必开 `resampling: smote` 就能拿到类别均衡的拟合。
`knn`/`nb`/`lda`/`gbdt`/`mlp`/`dnn` 没有 class-weight 参数——这些请配合
`resampling: smote` 使用。

| `model.name` | 底层估计器 | 默认值（YAML 可覆盖） | 类别均衡 | 额外依赖 |
|---|---|---|---|---|
| `rf` | `RandomForestClassifier` | `n_estimators=200`, `class_weight="balanced_subsample"`, `n_jobs=-1` | 自带 | 无 |
| `svm` | `SVC` | `C=1.0`, `kernel="rbf"`, `gamma="scale"`, `class_weight="balanced"`, `probability=True` | 自带 | 无 |
| `xgb` | `BalancedXGBClassifier`（自动类别加权） | — | 自带 | **xgboost**（在 `experiments` 组内） |
| `lgb` | `BalancedLGBMClassifier` | — | 自带 | **lightgbm**（不在组内，需 `pip install lightgbm`） |
| `catboost` | `BalancedCatBoostClassifier` | — | 自带 | **catboost**（不在组内，需 `pip install catboost`） |
| `knn` | `KNeighborsClassifier` | `n_neighbors=5`, `weights="distance"`, `n_jobs=-1` | 无——用 `resampling: smote` | 无 |
| `lr` | `LogisticRegression` | `C=1.0`, `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=1000` | 自带 | 无 |
| `lasso` | `LogisticRegression`（`penalty="l1"`, `solver="saga"`） | L1 正则 LR，`C=1.0`, `class_weight="balanced"`, `max_iter=1000` | 自带 | 无 |
| `elasticnet` | `LogisticRegression`（`penalty="elasticnet"`, `solver="saga"`） | 弹性网 LR，`l1_ratio=0.5`, `C=1.0`, `class_weight="balanced"`, `max_iter=1000` | 自带 | 无 |
| `nb` | `GaussianNB` | — | 无——用 `resampling: smote` | 无 |
| `lda` | `LinearDiscriminantAnalysis` | `solver="svd"` | 无——用 `resampling: smote` | 无 |
| `cart` | `DecisionTreeClassifier` | `class_weight="balanced"` | 自带 | 无 |
| `mlp` | `MLPClassifier` | `hidden_layer_sizes=(100,)`, `activation="relu"`, `solver="adam"`, `max_iter=300` | 无——用 `resampling: smote` | 无 |
| `dnn` | `MLPClassifier`（更深层） | `hidden_layer_sizes=(100,100,100)`, `solver="adam"`, `learning_rate="adaptive"`, `learning_rate_init=0.001`, `max_iter=300`, `early_stopping=True` | 无——用 `resampling: smote` | 无 |
| `extratrees` | `ExtraTreesClassifier` | `n_estimators=200`, `class_weight="balanced_subsample"`, `n_jobs=-1` | 自带 | 无 |
| `gbdt` | `GradientBoostingClassifier` | `n_estimators=100`, `learning_rate=0.1`, `max_depth=3` | 无——用 `resampling: smote` | 无 |

上表以外的名字（加上多视图场景的 `block_plsda`/`block_splsda`/`mogonet`/
`mogonet_nn`/`mofa`/`diablo`/`snf`——单视图实验用不到，需要的话看
`experiments/classifiers.py`）都会被 **`ValueError` 拒绝**，错误信息会
列出所有支持的名字——打错字不会静默跑。

## 传额外构造器参数

无论哪一层，YAML `model:` 下除 `name` 以外的键都会直接传给估计器构造器。
例如——更大的随机森林：

```yaml
model:
  name: rf
  n_estimators: 500
  max_depth: 10
  random_state: 7   # 所有模型都从 model.random_state 读取，默认 42
```

## `--grid` 与基线

`--grid` 默认只扫包内估计器的网格（`TBLS_GRID`/`BLS_GRID`）。**基线没有
默认网格**：对 `model.name: lr` 传 `--grid` 会退化为单次 k-fold 并打告警
（不会去扫假网格）。要真扫基线，在 YAML 里显式写 `grid:` 段：

```yaml
model: {name: lr}
grid:
  C: [0.01, 0.1, 0.1, 1.0, 10.0]
```

完整网格规则见 [grid-search.md](grid-search.md)。
