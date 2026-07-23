[English](./experimental-modules.md) | 简体中文

# `tbls.genoptim` 与 `tbls.ensemble`（实验性）

两个子包都随 `tbls` wheel 发布（无需 numpy/scipy/scikit-learn 之外的依赖），但属于**实验性**：导入任一子包都会发出 `FutureWarning`，其公开接口在次版本间可能不经通知即变更，亦不遵循核心估计器（`TBLS`、`BroadLearningSystem`、`PairwiseKCCA`、`GraphFuzzyKCCA`）所遵循的弃用政策。

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)  # 确认知悉其实验性状态
    from tbls.ensemble import TreeSelector, diversity_score
    from tbls.genoptim import ChromosomeEncoder, PopulationInitializer
```

## `tbls.ensemble`--功能完备

两个独立组件，与 `TBLS` 内部无耦合：

### `diversity_metrics`

```python
from tbls.ensemble import diversity_score, jaccard_similarity

feature_sets = [{1, 2, 3}, {2, 3, 4}, {5, 6}]
diversity_score(feature_sets, method="jaccard")   # 平均成对 (1 - Jaccard 相似度)
diversity_score(feature_sets, method="entropy")   # 特征出现频率的香农熵
```

### `TreeSelector`

针对 `{index: fitness_score}` 字典的通用 top-k / 阈值选择器--并不知晓也不关心索引来自 `TBLS` 的树；可配合你提供的任意适应度/多样性评分使用：

```python
from tbls.ensemble import TreeSelector

selector = TreeSelector(selection_method="top_k", weight_method="performance")
selector.fit(fitness_scores={0: 0.8, 1: 0.6, 2: 0.9, 3: 0.4})
selector.get_selected_trees()   # 所选子集的索引
selector.get_weights()          # 所选子集的归一化权重
```

## `tbls.genoptim`--部分可用

| 组件 | 状态 |
|---|---|
| `ChromosomeEncoder`、`PopulationInitializer` | 可用。纯粹的编码/解码与自举种群工具，与 `TBLS` 无耦合。 |
| `operators.selection` / `crossover` / `mutation` | 可用。作用于普通数组的标准 GA 算子（锦标赛/轮盘赌选择、均匀/单点交叉、位翻转/高斯/自适应变异）。 |
| `fitness.MultiObjectiveFitness` | **未针对当前 `TBLS` 验证。** |
| `ga_optimizer.GeneticOptimizer` | **未针对当前 `TBLS` 验证。** |

### 为何 `fitness.py`/`ga_optimizer.py` 无法用于 `tbls.tbls.TBLS`

这两个模块是针对一个已被移除的旧估计器（`TreeBroadLearningSystem`）编写的，调用了当前 `tbls.tbls.TBLS` 上不存在的属性：

| `genoptim` 中的调用 | 旧版 `TreeBroadLearningSystem` 是否存在 | 当前 `tbls.tbls.TBLS` 是否存在？ |
|---|---|---|
| `model.predict(X, trees=selected_trees)` | 是 | **否**--`predict(X)` 无 `trees=` 关键字参数 |
| `model.mapping_trees` | 是 | **否**--对应属性为 `map_trees_` |
| `tree.selected_features` | 是 | **否**--对应属性为 `RegressionTreeModule.feature_indices_` |
| `model.tree_params["bootstrap_ratio"]` | 是 | **否**，无此属性 |
| `model.n_map_nodes` | 是 | **否**--对应构造参数为 `n_map_trees` |
| `model.X_original` | 是 | **否**，无此属性 |

以 `tbls.tbls.TBLS` 实例调用 `MultiObjectiveFitness.calculate(model, ...)` 或 `GeneticOptimizer.optimize(model, ...)`，将在首次触及上述访问时抛出 `AttributeError`/`TypeError`。这**不是应静默打补丁的缺陷**--它反映了一种真实的能力缺口：当前 `TBLS` 没有"仅使用树的子集进行预测"的概念，也不具备 `genoptim` 所期望形态的逐树特征索引内省。

### 修复所需的工作

使 `genoptim` 适用于当前 `TBLS` 需要决策并实现新的 `TBLS` 能力，而非仅重命名属性：

1. 一种接受树子集（或每棵树的权重）的 `predict`/`predict_proba` 变体，而非始终使用完整的映射+增强集成。
2. 决定树选择作用于映射树、增强树还是二者兼有，以及所选子集的输出如何重组以进行输出权重求解（`TBLS` 当前是在完整堆叠特征矩阵 `A` 上联合训练 `W_`，而非逐树）。
3. 据所得 API 更新 `fitness.py`/`ga_optimizer.py`，并添加一个真正对拟合后的 `TBLS` 运行 `GeneticOptimizer.optimize(...)` 并对结果断言的端到端测试--在该测试存在之前，不要假定对这些模块的任何改动可用。

这刻意不在当前版本范围内--它属于新的估计器功能，而非打包/重构任务。

## 若现在就想使用遗传式树选择

在上述缺口弥合之前，请直接使用 `tbls.ensemble` 的独立组件，配合你自己的树子集/预测逻辑（例如自行存储逐树预测，并在 `ChromosomeEncoder` 解码出的掩码下组合），而非经由 `genoptim.fitness`/`ga_optimizer`。
