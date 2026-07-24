[English](./index.md) | 简体中文

# experiments 流水线 — 总览

`experiments/` 是本仓库的训练、评估与对比流水线。它在真实数据集上跑
`tbls` 估计器(以及基线),输出结构化日志 + Excel + 图表,并跨多次运行做
横向对比。它**不属于**已发布的 `tbls` PyPI 包(依赖更重),所以用户用
`uv sync --group experiments` 单独安装。

`docs/experiments/` 目录是这条流水线唯一的文档入口。它被拆成多篇,读者
既能快速浏览下方的[5 步快速上手](#5-步快速上手),也能在需要某一块细节时
直接跳到对应主题页。

## 各篇索引

| 想了解 | 阅读 |
|---|---|
| 把数据集放好并跑一次实验 | [datasets.md](datasets.md) |
| 写一次运行的 YAML(每个键、每个值) | [config-reference.md](config-reference.md) |
| 选模型名(`tbls`/`bls`/`lr`/`rf`/...) | [models.md](models.md) |
| 跑 `train.py`(每个 `--option` 干啥、有无冲突) | [cli-train.md](cli-train.md) |
| 从一次或多次运行出图 | [cli-visualize.md](cli-visualize.md) |
| 出跨运行对比 Excel | [cli-compare.md](cli-compare.md) |
| 用 `--grid` 扫参 | [grid-search.md](grid-search.md) |
| 跑完后磁盘上有什么(Excel sheet、JSONL 事件、npz) | [outputs.md](outputs.md) |
| 为啥 PR 图看起来有悬崖 | [figures-and-calibration.md](figures-and-calibration.md) |
| 改/扩展流水线本身 | [internals.md](internals.md) |

## 5 步快速上手

假设你已经 clone 仓库并准备好符合 [pkl 合同](datasets.md) 的数据集。所有
命令都在仓库根目录执行。

### 1. 装 experiments 环境

```bash
uv sync --group experiments
```

这会安装流水线用到的所有重依赖(`pandas`、`imbalanced-learn`、`openpyxl`、
`loguru`、`matplotlib`、`typer`、`pyyaml`、`xgboost`)。

### 2. 把数据集放到配置期望的位置

```bash
cp experiments/datasets/biomedical_larger.pkl examples/datasets/
```

示例配置读 `examples/datasets/{dataset}.pkl`;数据集目录被 git 忽略(大体积
二进制)。你的 pkl 必须满足什么:[datasets.md](datasets.md)。

### 3. 选/写一份配置 — 从示例开始

现成的消融配置在 `examples/configs/`:

```bash
ls examples/configs/
# tbls_plain.yaml  tbls_ifs.yaml  tlbs_graph.yaml  tbls_full.yaml  lr_baseline.yaml
```

每份都是一次运行的 YAML;打开任一份,按需改 `dataset:` / `model.name:` /
`preprocess:`(每个键都文档化在 [config-reference.md](config-reference.md))。
最简的一份是:

```yaml
dataset: biomedical_larger
data_path: examples/datasets/
run_name: My run
model: {name: tbls, use_if_weights: true, graph_gamma: 0.1}
preprocess: {feature_selection: lasso, resampling: smote}
cv: {n_splits: 5, random_state: 42}
output_dir: examples/runs
```

`run_name` 既作为运行目录的名称段,也作为图表/Excel 的图例标签,所以你写
什么就是什么,而**不是**自动生成的 `tbls_biomedical_larger/timestamp`。

### 4. 跑(单个配置,或批处理一整个目录)

```bash
# 一份配置
uv run --group experiments python experiments/train.py \
    --config examples/configs/tbls_full.yaml

# 目录下每个 *.yaml,都用 --n-splits 覆盖
uv run --group experiments python experiments/train.py \
    --config-dir examples/configs --n-splits 2
```

磁盘上会落:运行目录 + 各 cohort 的 Excel + JSONL + (非网格运行时)原始各
折预测。完整布局见 [outputs.md](outputs.md)。

### 5. 出图、出对比 Excel

```bash
# 图表 — 给 run-name 目录;CLI 自动找最新的 timestamp 子目录
uv run --group experiments python experiments/visualize.py \
    --dir "examples/runs/TBLS Full" \
    --dir "examples/runs/Logistic Regression" \
    --output-dir examples/plots --dpi 300

# 均值 ± 标准差 的对比 Excel,每个指标最优值加粗
uv run --group experiments python experiments/compare.py \
    --dir "examples/runs/TBLS Full" \
    --dir "examples/runs/Logistic Regression" \
    --output-dir examples/comparison
```

`--dir` 既可以接 run-name 层(自动选最新的 `YYYYMMDD_HHMMSS` timestamp
子目录),也可以接 run-name/timestamp 层(直接使用)。更深、更浅、或非
timestamp 的路径都会报错 — 不需要 shell 通配。完整规则见
[cli-visualize.md](cli-visualize.md) 和 [cli-compare.md](cli-compare.md)。

## 本文不涵盖

- `tbls`/`BroadLearningSystem` 估计器 API 本身:那是
  [`../usage-tbls.md`](../usage-tbls.md) 与 [`../usage-bls.md`](../usage-bls.md)。
- 多视图融合(pkl + CCA/GFCCA):那是
  [`../usage-multiview-fusion.md`](../usage-multiview-fusion.md),这里的
  `preprocess` / `fusion` 配置键只是把数据喂给它。
- 为啥 TBLS 的 PR 曲线看起来病态:
  [`figures-and-calibration.md`](figures-and-calibration.md)。
