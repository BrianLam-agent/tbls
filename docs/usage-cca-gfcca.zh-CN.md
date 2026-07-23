[English](./usage-cca-gfcca.md) | 简体中文

# 使用 `PairwiseKCCA` 与 `GraphFuzzyKCCA`

二者均为**双视图**核典型相关分析（CCA）特征提取器：给定两个从不同视角描述*相同*样本的特征矩阵（如两种模态、两个特征子集），它们在核空间中寻找最大化两视图相关性的投影方向，并可将任一视图的新样本投影至该共享空间。它们刻意不实现 sklearn 的 `TransformerMixin`，原因见 [`architecture.md`](./architecture.zh-CN.md)。

- `PairwiseKCCA`--普通正则化核 CCA。
- `GraphFuzzyKCCA`--额外引入直觉模糊集样本可信度评分与判别性图嵌入正则化项（与 `PairwiseKCCA` 不同，`fit` 时使用类别标签 `y`）。

## `PairwiseKCCA`

```python
import numpy as np
from tbls import PairwiseKCCA

X1_train, X2_train = np.random.randn(100, 30), np.random.randn(100, 20)
X1_test, X2_test = np.random.randn(20, 30), np.random.randn(20, 20)

cca = PairwiseKCCA(k=7, reg_lambda=0.1, kernel_gamma=0.1)
cca.fit(X1_train, X2_train)

Z1_train, Z2_train = cca.transform()          # 训练投影，两个视图
Z1_test = cca.transform_view1(X1_test)         # 投影新的视图 1 样本
Z2_test = cca.transform_view2(X2_test)         # 投影新的视图 2 样本
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `k` | `7` | 保留的典型变量对数。 |
| `reg_lambda` | `0.1` | 加到每个视图核矩阵上的岭正则化（用于数值稳定性）。 |
| `kernel_gamma` | `0.1` | 传给 `tbls._kernel.rbf_kernel` 的基础 RBF 核宽度（仍按成对距离中位数自适应缩放）。 |

## `GraphFuzzyKCCA`

同为双视图形态，但 `fit` 额外接受类别标签 `y`，并构建判别性图（`Lw - beta * Lb`）与 IFS 可信度权重：

```python
from tbls import GraphFuzzyKCCA

gfcca = GraphFuzzyKCCA(k=7, reg_lambda=0.1, kernel_gamma=0.1, graph_gamma=0.5)
gfcca.fit(X1_train, X2_train, y_train)

Z1_train, Z2_train = gfcca.transform()
Z1_test = gfcca.transform_view1(X1_test)
Z2_test = gfcca.transform_view2(X2_test)
```

`PairwiseKCCA` 之外的关键参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `graph_gamma` | `0.1` | 判别性图嵌入项的权重。 |
| `sigma_if` / `delta_if` | `1.0` / `0.5` | IFS 隶属度宽度 / 相对邻域阈值。 |
| `min_weight` | `1e-4` | IFS 权重最小裁剪值（防止加权核矩阵奇异）。 |
| `discriminative_beta` | `0.3` | 判别性图中类间惩罚权重。 |
| `max_attempts` / `tau_factor`（传给 `fit`） | `5` / `10.0` | 当广义特征问题非正定时，`fit` 最多重试 `max_attempts` 次，每次将数值稳定项放大 `tau_factor` 倍，直至放弃并抛出异常。 |

## 多视图特征构建流水线

对于两个以上视图，可使用模块级辅助函数：它们对视图的每个两两组合运行 CCA 并拼接结果：

```python
from tbls.cca import build_cca_features, project_cca_features
# 或：from tbls.gfcca import build_gfcca_features（需 y）；project_cca_features
#     刻意不从 gfcca 再导出--见 architecture.md。

X_views_train = [X1_train, X2_train, X3_train]
F_train, cca_models = build_cca_features(X_views_train, cca_k=7)

X_views_test = [X1_test, X2_test, X3_test]
F_test = project_cca_features(X_views_test, cca_models)
```

`cca_models` 将每个 `(i, j)` 视图对索引映射到其拟合好的 `PairwiseKCCA`（或经 `tbls.gfcca.build_gfcca_features` 得到的 `GraphFuzzyKCCA`）实例，使 `project_cca_features` 能为留出数据精确复现*训练时*的投影--训练集与测试集之间无数据泄漏。

## 不支持 `Pipeline`（设计如此）

由于这两个类在 `fit` 时需要两个对齐的特征矩阵、在 `transform` 时需指明"哪个视图"，它们无法在不静默丢弃某个视图的前提下满足 `sklearn.pipeline.Pipeline` 的单 `X` 契约。若需要，可自行编写小适配器，例如：

```python
from sklearn.base import BaseEstimator, TransformerMixin

class SingleViewCCA(BaseEstimator, TransformerMixin):
    """将 PairwiseKCCA 适配为单堆叠视图的 Pipeline 步骤。"""

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

该适配器未随 `tbls` 提供，因为"将两个视图堆叠为一个 `X`"的约定属于应用特定的选择（列范围、或元组、或 `dict`），并非库可通用决定之事。
