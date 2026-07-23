[English](./development.md) | 简体中文

# 开发指南

本指南即"二次开发教程"--涵盖本地开发环境搭建、运行与 CI 一致的检查，以及正确扩展本包所需的一切。

## 1. 环境搭建

```bash
git clone https://github.com/BrianLam-agent/tbls.git
cd tbls
uv sync --group dev --group experiments   # 开发工具 + experiments/ 依赖
```

`dev` 是一个 `uv` 依赖组（`pytest`、`pytest-cov`、`ruff`、`mypy`），在运行 `uv run ...` 时默认同步，故很少需要显式传 `--group dev`。`experiments` 并非默认组，必须显式请求（`uv sync --group experiments` 或 `uv run --group experiments ...`），因为它会拉取仅训练流水线所需的 `pandas`/`xgboost` 等。

## 2. 日常命令

```bash
uv run pytest tests/ -v                 # 完整测试套件
uv run pytest tests/test_tbls.py -v     # 单个文件
uv run ruff check .                     # lint
uv run ruff format .                    # 格式化（就地修改）
uv run ruff format --check .            # 格式化（仅检查，CI 运行此项）
uv run mypy src/tbls                    # 类型检查（严格，仅限包内）
```

pre-commit 钩子（`.pre-commit-config.yaml`）会自动运行 `ruff --fix`、`ruff-format`、`pyproject-fmt` 与 `yamlfmt`：

```bash
uv run pre-commit install     # 一次性，安装 git 钩子
uv run pre-commit run --all-files
```

## 3. 项目约定

- **文档字符串**：Google 风格、英文，位于每个公开类/函数上。由 ruff 的 `D` 规则组强制（`.ruff.toml` 中 `convention = "google"`）--`ruff check .` 会对缺失/格式错误的文档字符串报错。
- **类型提示**：`src/tbls/` 中每个公开签名需有完整类型提示；`mypy --strict` 必须通过。`experiments/` 不设同等门槛（它是内部工具，非发布 API）。
- **估计器契约**：新增到 `src/tbls/` 的任何分类器必须是完整的 `sklearn.base.BaseEstimator` + `ClassifierMixin`（具体清单见 [`architecture.md` 第 5 节](./architecture.zh-CN.md)）。新增的特征提取器默认应采用标准单 `X` 的 `fit`/`transform`（`TransformerMixin`），*除非*它本质上是多视图的（如 `PairwiseKCCA`/`GraphFuzzyKCCA`）--后两者是刻意的例外，而非可随意效仿的先例，原因见该节。
- **注释**：英文。
- **提交信息**：[Conventional Commits](https://www.conventionalcommits.org/)（`feat:`、`fix:`、`docs:`、`test:`、`build:`/`ci:`、`chore:`、`refactor:`、`perf:`、`revert:`）。这不仅是风格--`cliff.toml` 据此前缀自动构建变更日志；不合规范的提交信息会被静默排除在变更日志之外（见 [`release-process.md`](./release-process.zh-CN.md)）。

## 4. 向 `src/tbls` 新增估计器或特征

1. 以独立模块实现（`src/tbls/<name>.py`），复用 `tbls._kernel`/`tbls._ifs`/`tbls._graph`，而非重新推导 RBF 核/IFS/图的数学--见 [`architecture.md` 第 4 节](./architecture.zh-CN.md)。
2. 满足估计器契约（上文第 3 节）。
3. 从 `src/tbls/__init__.py` 的 `from .<module> import ...` 导出，并加入 `__all__`。
4. 添加测试：至少包括在合成数据上的 `fit`/`predict`/`predict_proba`（见 `tests/conftest.py` 的 fixture）、`sklearn.base.clone()` 往返、以及 `cross_val_score`/`GridSearchCV` 冒烟测试（参照既有 `tests/test_tbls.py`/`test_bls.py`）。若新代码涉及 `_kernel`/`_ifs`/`_graph`，在 `tests/test_shared_modules.py` 中添加**直接**单元测试以校验数值输出，而非仅看形状/有限性--原因见 [`architecture.md` 第 4 节"关于数值保真度"](./architecture.zh-CN.md)。
5. 编写文档：新增 `docs/usage-<name>.md`（含 `English | [简体中文](...)` 头部行），并在根 `README.md` 的文档索引中链接。
6. 在提 PR 前运行完整检查套件（第 2 节）。

## 5. 文档结构与翻译

`docs/` 下每份文档（以及根 `README.md`）均以如下行开头：

```markdown
English | [简体中文](./<同名>.zh-CN.md)
```

英文文档为权威来源，随代码变更同步维护。简体中文翻译（`*.zh-CN.md`）另行维护--若你新增或重构英文文档，请添加指向（可能尚未存在的）`.zh-CN.md` 对应文件的头部行，但不强制你亲自撰写翻译。

## 6. `experiments/` 与 `src/tbls/` 之分

若你的改动关乎在真实数据上*训练/评估*（新的数据加载器、新的对比分类器、新的命令行标志），应归入 `experiments/` 而非 `src/tbls/`--见 [`architecture.md` 第 3 节](./architecture.zh-CN.md)。若不确定归属，可自问："刚执行 `pip install tbls` 的人需要它吗？"若否，则属 `experiments/`。

## 7. 发布

不属于日常开发--若你拥有维护者权限且需要发布版本，见 [`release-process.md`](./release-process.zh-CN.md)。
