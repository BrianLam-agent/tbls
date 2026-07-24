[English](./architecture.md) | 简体中文

# 架构与仓库结构

本文档描述 `tbls` 仓库的组织方式、如此组织的缘由，以及塑造公开 API 的设计决策。它是仓库结构的权威来源；若下方的目录树与现实不符，以仓库为准，并提 issue/PR 修正本文档。

## 1. 仓库布局

```
tbls/                            # 仓库根目录
├── pyproject.toml                # 包元数据、依赖组、工具配置
├── README.md                     # 入口：安装、快速上手、文档索引
├── LICENSE                       # Apache-2.0
├── cliff.toml                    # git-cliff 变更日志配置
├── .ruff.toml                    # lint/格式配置
├── .pre-commit-config.yaml
├── .github/workflows/            # CI、发布、变更日志自动化
│
├── docs/                         # 本文档集
│
├── src/tbls/                     # 发布包（PyPI: `tbls`）
│   ├── __init__.py                # 公开 API
│   ├── py.typed                   # PEP 561 标记（包附带类型提示）
│   ├── _kernel.py                 # 共享：RBF 核工具
│   ├── _ifs.py                    # 共享：直觉模糊集评分
│   ├── _graph.py                  # 共享：图拉普拉斯构造
│   ├── bls.py                     # BroadLearningSystem 估计器
│   ├── tbls.py                    # TBLS 估计器
│   ├── cca.py                     # PairwiseKCCA + 特征构建流水线
│   ├── gfcca.py                   # GraphFuzzyKCCA + 特征构建流水线
│   ├── genoptim/                  # [实验性] 用于树选择的遗传优化器
│   └── ensemble/                  # [实验性] 树集成多样性/选择
│
├── experiments/                  # 训练命令行 + 数据流水线（不发布至 PyPI）
│   ├── configs/default.yaml
│   ├── datasets/                  # 真实 .pkl 数据集存放于此（git 忽略）
│   ├── dataprocess.py             # DataLoader：特征选择 + 重采样
│   ├── evaluate.py                # TBLSEvaluator + TBLSResultSaver
│   ├── classifiers.py             # 对比算法工厂
│   ├── train.py                   # typer 命令行入口
│   └── smoke_run.py               # 最小真实数据集健全性检查
│
└── tests/                        # 面向发布包的 pytest 测试套件
```

## 2. 为何采用 `src/` 布局

采用 `src/tbls/`（而非顶层 `tbls/`）可避免在未安装的情况下误导入仓库内的包--每个测试与脚本都针对*已安装*的包运行，从而在抵达用户之前捕获打包缺陷（缺失 `__init__.py`、`pyproject.toml` 中 `packages =` 通配有误等）。这是 Python 打包权威机构对可发布包推荐的布局。

## 3. 为何区分包与实验代码

本仓库实现了一种研究方法（TBLS，及其所基于的 BLS/CCA/GFCCA 构件），但有两类截然不同的使用者：

- **库使用者**（`pip install tbls`）希望得到一个小巧、稳定、依赖轻量的包：仅 `numpy` + `scipy` + `scikit-learn`。他们通过标准的 `fit`/`predict`/`predict_proba` 契约使用估计器，可能置于 `sklearn.pipeline.Pipeline` 或 `GridSearchCV` 之内。
- **本仓库自身的实验**则需要一套更重、更带主观取向的技术栈：`pandas`、`imbalanced-learn`、`xgboost`、`typer`、`pyyaml`、`openpyxl`、命令行、数据集加载器、Excel 报告写入器，以及一个庞大的对比分类器工厂（`experiments/classifiers.py`）。

若将后一套技术栈的依赖随 PyPI 包发布，将迫使每位库使用者为其从不调用的代码安装 `xgboost`/`pandas` 等。因此 `experiments/` 置于 `src/tbls/` 之外，绝不进入 wheel（由 `pyproject.toml` 的 `[tool.hatch.build.targets.wheel] packages = ["src/tbls"]` 保证--可用 `unzip -l dist/*.whl` 验证），其依赖为可选的 `uv` 依赖组（`uv sync --group experiments`），不属于 `[project.dependencies]`。

## 4. 包内共享模块

`TBLS`、`PairwiseKCCA` 与 `GraphFuzzyKCCA` 都需计算 RBF 核矩阵，而 `TBLS`/`GraphFuzzyKCCA` 都需计算直觉模糊集（IFS）样本可信度评分。这些在过去是三份独立、相互复制的实现。现已合并为三个以下划线为前缀的私有模块，由估计器导入但不属于公开 API：

| 模块 | 用途 | 使用者 |
|---|---|---|
| `tbls._kernel` | `rbf_kernel`（通用，`gamma` 由调用方提供）与 `compute_kernel_matrix`/`kernel_distance_matrix`（TBLS 的自适应 gamma 变体） | `tbls.py`、`cca.py`、`gfcca.py` |
| `tbls._ifs` | `compute_if_scores_geib`（GEIB 公式，对角矩阵）与 `compute_if_scores_simple`（隶属度/非隶属度/犹豫度，向量） | `tbls.py`、`gfcca.py` |
| `tbls._graph` | `build_graph_laplacian`（kNN 内在/惩罚）与 `build_discriminative_graph_laplacian`（仅基于标签的 `Lw - beta*Lb`，移植自 `GraphFuzzyKCCA` 的内联副本--刻意未与 `gfcca.py` 去重） | `tbls.py` |

这三个模块**刻意不**从 `tbls/__init__.py` 再导出--它们是实现细节，可在不改变公开 API 的前提下被重构（例如以 Cython 扩展替换 `rbf_kernel`/图构造的热点路径）。若你要扩展 `tbls`，请优先调用这些模块，而非第四次重复核/IFS/图的数学推导；参见 [`development.md`](./development.zh-CN.md)。

### 关于数值保真度

这三个模块抽取自三份原先相互独立的实现，因此对它们进行了直接单元测试（`tests/test_shared_modules.py`），对照从零独立实现的参考计算，而不仅检查形状/有限性。这一点切中要害：此前一次重构引入了 `build_graph_laplacian` 相似度带宽的静默回归（仅取 kNN 选中边的中位数，而非*所有*成对距离的中位数，从而改变了拟合后的正则化强度）。仅检查形状/有限性的测试未能发现；而与独立参考实现的逐位比对则发现了。若你修改 `_kernel.py`、`_ifs.py` 或 `_graph.py`，请保留（或扩展）此类测试。

## 5. 估计器契约

`tbls` 中发布的每个分类器（`TBLS`、`BroadLearningSystem`）都是完整的 scikit-learn 估计器：

- 继承 `sklearn.base.BaseEstimator` + `ClassifierMixin`。
- 每个构造函数参数都以同名属性存储（`sklearn.base.clone()` 的要求）--`get_params`/`set_params` 继承自 `BaseEstimator`，而非手写。
- 实现 `fit(X, y) -> self`、`predict(X)`、`predict_proba(X)`。
- 在 `fit` 中设置 `self.classes_` / `self.n_classes_`。
- 在 `fit`/`predict` 中不做文件 I/O、不记录日志--训练/评估的副作用（Excel 报告、进度条）位于 `experiments/`，不在包内。

`PairwiseKCCA` 与 `GraphFuzzyKCCA` 是特征提取器而非分类器，且具有**双视图**接口，不符合 sklearn 单参数 `TransformerMixin.transform(X)` 契约：

```python
model.fit(X1, X2)                 # PairwiseKCCA；GraphFuzzyKCCA 还需传入 y
model.transform_view1(X1_new)     # 投影新的视图 1 样本
model.transform_view2(X2_new)     # 投影新的视图 2 样本
model.transform()                 # 无参：返回两个视图的*训练*投影
```

它们仅继承 `BaseEstimator`（用于 `get_params`/`set_params`/`clone()`），**刻意不**继承 `TransformerMixin`--双参数的 `fit(X1, X2)` 与需要指明两个特征矩阵之一的 `transform()`，无法在不静默丢弃某个视图或不发明一种仍不满足 `sklearn.pipeline.Pipeline` 的非标准调用约定的前提下塞入 `transform(X)`。若你需要为双视图模型获得 `Pipeline` 兼容性，请自行编写适配器（例如一个将两视图拼接或打包为元组的小适配器），而非期望 `tbls` 直接提供。参见 [`usage-cca-gfcca.md`](./usage-cca-gfcca.zh-CN.md)。

## 6. 实验性子包：`tbls.genoptim`、`tbls.ensemble`

`tbls.genoptim`（用于选择/加权 `TBLS` 树的遗传优化器）与 `tbls.ensemble`（树多样性度量与通用的 top-k/阈值选择器）随包发布（它们无需 numpy/scipy/scikit-learn 之外的依赖，故无打包上的理由将其排除），但明确标注为实验性：

- 每个子包的 `__init__.py` 在首次导入时发出 `FutureWarning`。
- `tbls.ensemble` 与 `TBLS` 内部无耦合，功能完备。
- `tbls.genoptim.fitness`/`ga_optimizer` 引用了当前 `tbls.tbls.TBLS` 上**并不存在**的属性（`mapping_trees`、`tree.selected_features`、`predict` 的 `trees=` 关键字参数）--它们沿用自更早的估计器 API，且未经验证可端到端运行。完整原委及修复所需的工作量见 [`experimental-modules.md`](./experimental-modules.zh-CN.md)。

## 7. 数据流（实验代码）

```
experiments/datasets/*.pkl
        │  joblib.load
        ▼
experiments/dataprocess.py::DataLoader   (特征选择 + 重采样，
        │                                 仅在训练折上拟合)
        ▼
tbls.TBLS.fit(X_train, y_train)
        │
        ▼
experiments/evaluate.py::TBLSEvaluator   (sklearn 指标：accuracy、F1、
        │                                 AUROC、balanced accuracy 等)
        ▼
experiments/evaluate.py::TBLSResultSaver (写入 results_dir/.../*.xlsx)
```

`experiments/train.py` 将其组织为由 YAML 配置（`experiments/configs/default.yaml`）驱动、支持 typer 命令行覆盖的 k 折交叉验证循环。参见 [`usage-experiments-cli.md`](./usage-experiments-cli.zh-CN.md) 获取完整 CLI/YAML/产物参考，参见 [`usage-figures-and-calibration.md`](./usage-figures-and-calibration.zh-CN.md) 了解为何 `TBLS` 出现颠倒的 PR 图（未标定 ridge 输出的 `0.5` 得分密庅）及未来标定工作如何解决。

## 8. 发布工程

版本管理、变更日志与 PyPI 发布流水线 documented 于 [`release-process.md`](./release-process.zh-CN.md)。简言之：`master` 上的 Conventional Commits -> `git-cliff`（配置：`cliff.toml`）在推送标签时将提交历史转为变更日志 -> GitHub Actions 构建 wheel/sdist，创建附带变更日志与两个产物的 GitHub Release，并经 Trusted Publishing（OIDC，不存储 API token）发布至 PyPI。

## 9. 后续阅读

- 初次接触本包？从根目录 [`README.md`](../README.zh-CN.md) 开始。
- 想使用某个估计器？见 `usage-tbls.md`、`usage-bls.md`、`usage-cca-gfcca.md`。
- 想用训练/可视化/对比命令行跑真实数据？见 `usage-experiments-cli.md`（完整 CLI + YAML + 产物参考）与 `usage-figures-and-calibration.md`（为何部分 `TBLS` PR 图看起来暴反）。
- 想贡献代码？见 [`development.md`](./development.zh-CN.md)。
- 想了解 `genoptim`/`ensemble` 的局限？见 [`experimental-modules.md`](./experimental-modules.zh-CN.md)。
- 想发布一个版本？见 [`release-process.md`](./release-process.zh-CN.md)。
