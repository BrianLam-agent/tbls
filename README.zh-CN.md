[English](./README.md) | 简体中文

# tbls

[![CI](https://github.com/BrianLam-agent/tbls/actions/workflows/ci.yml/badge.svg)](https://github.com/BrianLam-agent/tbls/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tbls.svg)](https://pypi.org/project/tbls/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/tbls.svg)](https://pypi.org/project/tbls/)

基于树的宽度学习系统（Tree-based Broad Learning System, TBLS），面向分类任务，提供完全兼容 scikit-learn 的接口。

`tbls` 打包了一个基于树的宽度学习系统分类器（`TBLS`）：由若干小型回归树按宽度学习系统的"映射 -> 增强"两阶段架构排列，并以闭式（伪逆）岭求解进行训练；同时附带其构成构件--经典随机权重 `BroadLearningSystem`，以及双视图核典型相关分析特征提取器（`PairwiseKCCA`、`GraphFuzzyKCCA`），后者支持可选的直觉模糊集样本加权与图拉普拉斯正则化。

## 安装

```bash
pip install tbls
```

发布版包仅依赖 `numpy`、`scipy` 与 `scikit-learn`。

## 快速上手

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from tbls import TBLS

X, y = make_classification(n_samples=200, n_features=20, random_state=0)

model = TBLS(n_map_trees=10, n_enhance_trees=10, random_state=0)
model.fit(X, y)
print(model.predict(X[:5]))
print(model.predict_proba(X[:5]))

print(cross_val_score(model, X, y, cv=3))
```

## 包内组件

| 组件 | 说明 |
|---|---|
| `tbls.TBLS` | 基于树的宽度学习系统分类器，支持可选的 IFS 样本加权与图拉普拉斯正则化。 |
| `tbls.BroadLearningSystem` | 经典随机权重宽度学习系统，具备真正的增量增强学习能力。 |
| `tbls.PairwiseKCCA` | 双视图正则化核典型相关分析特征提取器。 |
| `tbls.GraphFuzzyKCCA` | 双视图核典型相关分析，含 IFS 样本可信度与判别性图嵌入正则化。 |
| `tbls.genoptim` *（实验性）* | 遗传算法树选择--编码/算子可用，与 `TBLS` 耦合的适应度/优化器不可用（见文档）。 |
| `tbls.ensemble` *（实验性）* | 树多样性度量与通用的、基于适应度的树/子集选择器--功能完备。 |

所有分类器（`TBLS`、`BroadLearningSystem`）均为标准的 `sklearn.base.BaseEstimator`/`ClassifierMixin` 实现，可配合 `cross_val_score`、`GridSearchCV`、`Pipeline` 等使用。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/architecture.md`](docs/architecture.zh-CN.md) | 仓库结构、包与实验代码分离的缘由、共享模块设计、估计器契约。 |
| [`docs/usage-tbls.md`](docs/usage-tbls.zh-CN.md) | `TBLS` 教程：参数、IFS/图正则化、增量层、可复现性、性能说明。 |
| [`docs/usage-bls.md`](docs/usage-bls.zh-CN.md) | `BroadLearningSystem` 教程：参数、类别不平衡、Woodbury 增量增强。 |
| [`docs/usage-cca-gfcca.md`](docs/usage-cca-gfcca.zh-CN.md) | `PairwiseKCCA`/`GraphFuzzyKCCA` 教程：双视图接口、多视图流水线、为何不支持 `Pipeline`。 |
| [`docs/experiments/index.md`](docs/experiments/index.md) | `experiments/` 流水线一站式文档：5 步快速上手 + 分篇 — 数据集、YAML 配置参考、每个 `model.name`、三个 CLI、网格搜索、产物文件、TBLS PR 悬崖说明。 |
| [`docs/experimental-modules.md`](docs/experimental-modules.zh-CN.md) | `tbls.genoptim`/`tbls.ensemble` 中哪些可用、哪些不可用，以及原因。 |
| [`docs/development.md`](docs/development.zh-CN.md) | 本地开发环境、约定、如何新增估计器、文档/翻译结构。 |
| [`docs/release-process.md`](docs/release-process.zh-CN.md) | 语义化版本、变更日志生成、由标签触发的 CI/CD 发布流水线、PyPI 发布。 |

## 开发

```bash
uv sync --group dev --group experiments   # 安装全部依赖
ruff check .                              # 代码检查
ruff format --check .                     # 格式检查
mypy src/tbls                             # 类型检查
pytest                                     # 运行测试
```

完整指南见 [`docs/development.md`](docs/development.zh-CN.md)。

## 贡献

欢迎提交 issue 与 pull request。请先阅读 [`docs/development.md`](docs/development.zh-CN.md)，了解项目约定（Conventional Commits、估计器契约、文档/翻译结构）。

## 参考文献

`TBLS` 所采用的 GEIB 直觉模糊集公式遵循 Chen 等人发表于 *IEEE Transactions on Fuzzy Systems*（2025）的工作；宽度学习系统的整体架构则遵循原始 BLS 文献。

## 许可证

Apache License 2.0--详见 [LICENSE](LICENSE)。
