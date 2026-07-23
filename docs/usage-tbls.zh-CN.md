[English](./usage-tbls.md) | 简体中文

# 使用 `TBLS`

`tbls.TBLS` 是一个基于树的宽度学习系统分类器：由若干小型回归树按宽度学习系统的"映射 -> 增强"两阶段架构排列的集成，以闭式（伪逆）岭求解训练，而非梯度下降，并支持可选的直觉模糊集（IFS）样本加权与图拉普拉斯正则化。

## 安装

```bash
pip install tbls
```

## 快速上手

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from tbls import TBLS

X, y = make_classification(n_samples=300, n_features=20, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = TBLS(n_map_trees=20, n_enhance_trees=20, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # 形状 (n_samples, n_classes)
```

`TBLS` 适用于二分类与多分类问题；`y` 可为 `sklearn.preprocessing.LabelEncoder` 支持的任意标签类型（内部会做标签编码，`model.classes_` 保存原始标签，用于 `predict` 的输出）。

## 构造函数参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `n_map_trees` | `20` | 映射阶段回归树数量。每棵树在输入的 Poisson(1) 自举样本上、以随机子空间特征采样训练，其标量预测成为一个映射特征。 |
| `n_enhance_trees` | `20` | 增强阶段树数量，在*映射后*的特征（而非原始输入）上训练。 |
| `n_increment_layers` | `0` | 初始拟合后追加的增强层数（每层以扩充后的特征矩阵重新计算岭求解）。 |
| `tree_max_depth` | `5` | 每棵映射/增强树的最大深度。 |
| `tree_min_samples_split` | `3` | 每棵树节点分裂所需的最小样本数。 |
| `tree_max_features_ratio` | `0.7` | 每棵树采样的输入特征比例（随机子空间方法）。 |
| `reg_param` | `1e-4` | 输出权重求解的岭正则化强度。 |
| `use_if_weights` | `False` | 若为 `True`，按样本的直觉模糊集可信度评分对其加权（降低模糊/边界样本的权重）。 |
| `if_sigma` | `1.0` | IFS 评分的邻域半径尺度。 |
| `graph_gamma` | `0.0` | 图拉普拉斯正则化权重；`0` 完全禁用（甚至不构建图）。 |
| `graph_alpha_in` / `graph_alpha_p` | `1.0` / `1.0` | 内在（类内）与惩罚（类间）图拉普拉斯的相对权重。 |
| `graph_knn` | `10` | 构建相似图所用的最近邻数。`<= 0` 表示全连接。 |
| `use_kernel_for_graph` | `True` | 若为 `True`，图距离在 RBF 核空间而非原始欧氏空间中计算。 |
| `random_state` | `None` | 自举采样、特征子空间选择与树播种的随机种子。 |

`graph_threshold` 与 `class_sensitive` 为保留的构造参数，仅为兼容 `sklearn.base.clone()`/`get_params()` 而保留，当前在 `fit` 内部并不使用。

## 何时启用 IFS 加权与图正则化

二者均为可选，且带来额外计算开销（一个 `O(n²)` 的核/距离矩阵，IFS 还需 `O(n·k)` 的邻域循环）：

```python
model = TBLS(
    n_map_trees=20,
    n_enhance_trees=20,
    use_if_weights=True,      # 降低模糊/边界样本权重
    graph_gamma=0.1,          # 启用图拉普拉斯正则化
    graph_knn=10,
    random_state=0,
)
model.fit(X_train, y_train)
```

当类别在特征空间中严重重叠时（IFS 加权），或当你预期存在普通岭求解会忽略的有用的局部/全局类别结构时（图正则化），启用它们。对于干净、良好分离的合成数据，通常无明显差异--sklearn 兼容性测试套件主要将其用作数值保真度的回归检查（见 [`architecture.md`](./architecture.zh-CN.md)），而非因其总是带来收益。

## 增量层

```python
model = TBLS(n_map_trees=20, n_enhance_trees=20, n_increment_layers=2, random_state=0)
model.fit(X_train, y_train)  # 拟合基础模型，随后追加 2 个增强层
```

每个增量层新增 `n_enhance_trees` 棵增强树（在*当前*已扩展的特征矩阵上训练），并在扩充后的特征矩阵上重新计算岭求解。这以训练时间换取容量，且不丢弃已拟合的映射树。

## 配合 scikit-learn 工具使用

`TBLS` 是标准的 `BaseEstimator`/`ClassifierMixin`，可与 `cross_val_score`、`GridSearchCV`、`Pipeline` 等配合：

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

## 可复现性

`random_state` 控制 `fit` 内部的所有随机性来源（自举重采样、随机子空间特征选择、逐树种子）。两个 `TBLS` 实例在相同 `random_state` 与相同输入下产生完全相同的输出。

## 性能说明

- 训练开销随 `n_map_trees + n_enhance_trees * (1 + n_increment_layers)` 次回归树拟合增长，并在 `use_if_weights` 或 `graph_gamma > 0` 时额外有一次 `O(n²)` 的核/距离计算。
- 在新数据集上初次尝试时，宜从小规模起手（`n_map_trees=10, n_enhance_trees=10`、`use_if_weights=False`、`graph_gamma=0.0`）--这正是 `experiments/smoke_run.py` 为在数秒内健全性检查数据集所做的；见 [`usage-experiments-cli.md`](./usage-experiments-cli.zh-CN.md)。
- RBF 核与图拉普拉斯计算的复杂度在*训练*样本数上为 `O(n²)`；对极大数据集，在启用 `use_if_weights`/`graph_gamma` 前先做子采样。
