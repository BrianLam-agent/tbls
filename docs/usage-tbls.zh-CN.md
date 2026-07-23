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
| `if_sigma` | `1.0` | 高斯隶属度带宽，以数据中位成对欧氏距离为单位（非绝对尺度，与 `if_delta` 一致）。 |
| `graph_gamma` | `0.0` | 图拉普拉斯正则化权重；`0` 完全禁用（甚至不构建图）。 |
| `graph_alpha_in` / `graph_alpha_p` | `1.0` / `1.0` | 内在（类内）与惩罚（类间）图拉普拉斯的相对权重。 |
| `graph_knn` | `10` | 构建相似图所用的最近邻数。`<= 0` 表示全连接。 |
| `use_kernel_for_graph` | `True` | 若为 `True`，图距离在 RBF 核空间而非原始欧氏空间中计算。 |
| `random_state` | `None` | 自举采样、特征子空间选择与树播种的随机种子。 |
| `graph_strategy` | `"discriminative"` | 图拉普拉斯公式：`"discriminative"`（默认，`GraphFuzzyKCCA` 调优过的、仅基于标签的 `Lw - beta*Lb`）或 `"knn"`（原始 kNN 图）。 |
| `if_strategy` | `"simple"` | IFS 评分公式：`"simple"`（默认，`GraphFuzzyKCCA` 调优过的逐类中心 + 相对邻域）或 `"geib"`（原始 GEIB 公式）。 |
| `discriminative_beta` | `0.3` | `graph_strategy="discriminative"` 时的类间惩罚权重。 |
| `if_delta` | `0.5` | `if_strategy="simple"` 时的相对邻域阈值（以数据中位成对欧氏距离为单位）。 |
| `if_min_weight` | `1e-4` | `if_strategy="simple"` 时的 IFS 权重最小裁剪值。 |

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

## 图与 IFS 策略

`TBLS` 提供两种图拉普拉斯公式与两种 IFS 评分公式，可独立选择。默认值复现 `GraphFuzzyKCCA` 调优过的公式（同样的数学，已在 `tbls.gfcca` 中验证）；备选项则逐字复现 `TBLS` 在引入策略切换前的原始行为。

| 策略 | 默认值 | 备选项 |
|---|---|---|
| `graph_strategy` | `"discriminative"`--仅基于标签的判别图 `L = Lw - beta*Lb`（无 kNN、无距离），移植自 `GraphFuzzyKCCA`。 | `"knn"`--原始 kNN 图 `L = alpha_in*L_in - alpha_p*L_p`（`_graph.build_graph_laplacian`）。 |
| `if_strategy` | `"simple"`--逐类欧氏中心距离 + 相对邻域 IFS（`_ifs.compute_if_scores_simple`）。 | `"geib"`--原始的在核空间中的 GEIB 公式（`_ifs.compute_if_scores_geib`）。 |

```python
# 默认（调优过的 GFCCA 公式）：
TBLS(use_if_weights=True, graph_gamma=0.1)
# 原始 TBLS 行为：
TBLS(use_if_weights=True, graph_gamma=0.1, graph_strategy="knn", if_strategy="geib")
```

`discriminative_beta`、`if_delta` 与 `if_min_weight` 仅参数化默认（`"discriminative"`/`"simple"`）公式；当对应策略设为 `"knn"`/`"geib"` 时它们被忽略。不支持的策略字符串会在 `fit` 时抛出 `ValueError`。

## 消融变体（GTBLS/FTBLS/GFTBLS）

图项（`graph_gamma`，即"G"轴）与模糊 IFS 项（`use_if_weights`，即"F"轴）是两个相互独立的开关，因此四种消融组合均可直接在 `TBLS` 上实现：

| 变体 | `use_if_weights` | `graph_gamma` | 含义 |
|---|---|---|---|
| `tbls` | `False` | `0.0` | 两者皆无--朴素树 BLS。 |
| `gtbls` | `False` | `> 0` | 仅图正则化。 |
| `ftbls` | `True` | `0.0` | 仅模糊 IFS 样本加权。 |
| `gftbls` | `True` | `> 0` | 两者皆开（当前调优后的默认组合）。 |

用于消融研究时，`build_tbls_variant` 是一个轻便的工厂函数，按名称选取 `(use_if_weights, graph_gamma)` 组合，而非新增第三个可能与现有参数冲突的构造参数（那会令同一状态有两种表达方式，并使 `clone()`/`get_params()` 的往返复杂化）：

```python
from tbls import build_tbls_variant

gtbls = build_tbls_variant("gtbls", graph_gamma=0.2, n_map_trees=15)
ftbls = build_tbls_variant("ftbls", n_map_trees=15)
gftbls = build_tbls_variant("gftbls", graph_gamma=0.2, n_map_trees=15)
plain = build_tbls_variant("tbls", n_map_trees=15)
```

该工厂会原样转发其余 `TBLS` 构造关键字（`n_map_trees`、`graph_strategy`、`if_strategy`、`random_state` 等）。在以下情形它会抛出 `ValueError`：未知变体；对启用图的变体（`"gtbls"`/`"gftbls"`）传入 `graph_gamma <= 0`（非正值会静默禁用图项，使消融失去意义）；以及显式传入 `use_if_weights`（该工厂的全部意义即在于由 `variant` 明确设定此项）。

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
