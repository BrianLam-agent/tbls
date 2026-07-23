[English](./usage-bls.md) | 简体中文

# 使用 `BroadLearningSystem`

`tbls.BroadLearningSystem` 是经典的宽度学习系统分类器：随机权重特征映射节点、随机权重增强节点、以及岭伪逆输出层，支持可选的类别加权与基于 Woodbury 矩阵恒等式的真正增量增强学习（无需从头重训）。

## 快速上手

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

## 构造函数参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `n_feature_groups` | `10` | 独立的特征映射节点组数。 |
| `n_feature_nodes_per_group` | `100` | 每个特征映射组的节点数。 |
| `n_enhancement_groups` | `10` | 增强节点组数（在映射特征之上构建）。 |
| `n_enhancement_nodes_per_group` | `100` | 每个增强组的节点数。 |
| `map_func` | `"relu"` | 特征映射节点的激活函数：`relu`、`sigmoid`、`tanh`、`linear` 或 `leaky_relu`。 |
| `enhance_func` | `"relu"` | 增强节点的激活函数（同上可选项）。 |
| `reg_param` | `1e-8` | 伪逆求解的岭正则化参数。 |
| `class_weights` | `None` | `None`（不加权）、`"auto"`（经 `sklearn.utils.class_weight.compute_class_weight` 按类频率倒数加权）或显式的 `{class_label: weight}` 字典。 |
| `random_state` | `None` | 随机特征/增强权重矩阵的种子。 |

## 类别不平衡

```python
model = BroadLearningSystem(class_weights="auto", random_state=0)
# 或显式权重：
model = BroadLearningSystem(class_weights={0: 1.0, 1: 5.0}, random_state=0)
```

## 增量增强（Woodbury 更新）

与 `TBLS` 的增量层（重新计算完整岭求解）不同，`BroadLearningSystem.incremental_enhance` 以闭式更新伪逆，无需从头重解：

```python
model = BroadLearningSystem(random_state=0)
model.fit(X_train, y_train)

# 之后，在不丢弃已拟合输出权重的情况下增加容量：
model.incremental_enhance(X_train, num_new_nodes=100)
model.predict(X_test)
```

传入 `incremental_enhance` 的 `X` 必须是 `fit` 中所用的*同一*训练数据（或一致的替代批次）--它会为该数据重新计算既有的特征/增强节点以导出校正项；这并非用于在新样本上训练的机制。

## 缺失值

`fit`/`predict`/`predict_proba` 均在标准化前将输入经 `np.nan_to_num(..., nan=0.0)` 处理--`NaN` 值被静默替换为 `0.0`（概念上即标准化器见到它们之前）。若你的数据含有有意义的缺失值，请自行插补（如 `sklearn.impute.SimpleImputer`），而非依赖此回退。

## 配合 scikit-learn 工具使用

`BroadLearningSystem` 是标准的 `BaseEstimator`/`ClassifierMixin`，与 `cross_val_score`、`GridSearchCV`、`Pipeline` 等的配合方式与 `TBLS` 完全一致--见 [`usage-tbls.md`](./usage-tbls.zh-CN.md)。
