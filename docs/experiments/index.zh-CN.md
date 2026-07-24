English | [简体中文](./index.zh-CN.md)

# experiments 流水线 — 总览

`experiments/` 是本项目的训练、评估与对比流水线。它在真实数据集上跑 `tbls`
估计器（以及各种基线），输出结构化日志、Excel 和图表，并支持跨运行横向
对比。它**不属于**已发布的 `tbls` 包（依赖更重），需单独安装：

```bash
uv sync --group experiments
```

`docs/experiments/` 是这条流水线唯一的文档入口，按主题拆成多篇。想快速
上手看下面的[5 步速览](#5-步速览)，想深挖某个主题直接跳对应页面。

## 各篇索引

| 想了解 | 阅读 |
|---|---|
| 准备数据集并跑一次实验 | [datasets.md](datasets.md) |
| 写 YAML 配置（每个键的含义与取值） | [config-reference.md](config-reference.md) |
| 选模型（`tbls`/`bls`/`lr`/`rf`/...） | [models.md](models.md) |
| 跑 `train.py`（每个命令行选项） | [cli-train.md](cli-train.md) |
| 从运行结果出图 | [cli-visualize.md](cli-visualize.md) |
| 跨运行对比出 Excel | [cli-compare.md](cli-compare.md) |
| 用 `--grid` 做超参搜索 | [grid-search.md](grid-search.md) |
| 运行结束后磁盘上有什么 | [outputs.md](outputs.md) |
| PR 曲线为什么有悬崖 | [figures-and-calibration.md](figures-and-calibration.md) |
| 改/扩展流水线代码 | [internals.md](internals.md) |

## 5 步速览

假设你已 clone 仓库，数据集符合 [pkl 格式要求](datasets.md)。所有命令
在仓库根目录执行。

### 1. 安装环境

```bash
uv sync --group experiments
```

安装流水线所需的全部依赖（`pandas`、`imbalanced-learn`、`openpyxl`、
`loguru`、`matplotlib`、`typer`、`pyyaml`、`xgboost` 等）。

### 2. 放数据集

```bash
cp experiments/datasets/biomedical_larger.pkl examples/datasets/
```

示例配置从 `examples/datasets/{dataset}.pkl` 读取数据。数据集目录已
gitignore（大文件不提交）。pkl 格式要求详见 [datasets.md](datasets.md)。

### 3. 选一份配置

现成的消融配置在 `examples/configs/`：

```bash
ls examples/configs/
# tbls_plain.yaml  tbls_ifs.yaml  tbls_graph.yaml  tbls_full.yaml  lr_baseline.yaml
```

打开任一份按需修改即可。每个键的含义见
[config-reference.md](config-reference.md)。最简配置：

```yaml
dataset: biomedical_larger
data_path: examples/datasets/
run_name: My run
model: {name: tbls, use_if_weights: true, graph_gamma: 0.1}
preprocess: {feature_selection: lasso, resampling: smote}
cv: {n_splits: 5, random_state: 42}
output_dir: examples/runs
```

`run_name` 同时作为运行目录名和图表图例标签，你写什么就是什么——不是
自动生成的 `tbls_biomedical_larger/timestamp`。

### 4. 跑

```bash
# 单个配置
uv run --group experiments python experiments/train.py \
    --config examples/configs/tbls_full.yaml

# 批量跑整个目录，统一覆盖折数
uv run --group experiments python experiments/train.py \
    --config-dir examples/configs --n-splits 2
```

运行结束后磁盘上会有：运行目录、各 cohort 的 Excel、JSONL 日志、（非
网格搜索时）原始各折预测。完整布局见 [outputs.md](outputs.md)。

### 5. 出图、出对比表

```bash
# 出图——给 run-name 目录即可，自动选最新 timestamp
uv run --group experiments python experiments/visualize.py \
    --dir "examples/runs/TBLS Full" \
    --dir "examples/runs/Logistic Regression" \
    --output-dir examples/plots --dpi 300

# 出对比 Excel——均值 ± 标准差，最优值加粗
uv run --group experiments python experiments/compare.py \
    --dir "examples/runs/TBLS Full" \
    --dir "examples/runs/Logistic Regression" \
    --output-dir examples/comparison
```

`--dir` 可以给 run-name 层（自动选最新 `YYYYMMDD_HHMMSS` 子目录），
也可以给 run-name/timestamp 层（直接使用）。更深、更浅或非 timestamp
的路径都会报错——不需要 shell 通配。详见
[cli-visualize.md](cli-visualize.md) 和 [cli-compare.md](cli-compare.md)。

## 不在本文范围内

- `tbls`/`BroadLearningSystem` 估计器 API：见
  [`../usage-tbls.md`](../usage-tbls.md) 和 [`../usage-bls.md`](../usage-bls.md)。
- 多视图融合（pkl + CCA/GFCCA）：见
  [`../usage-multiview-fusion.md`](../usage-multiview-fusion.md)，这里的
  `preprocess`/`fusion` 配置键只是把数据喂给它。
- TBLS PR 曲线为什么看起来病态：见
  [`figures-and-calibration.md`](figures-and-calibration.md)。
