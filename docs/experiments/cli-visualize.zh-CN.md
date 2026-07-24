English | [简体中文](./cli-visualize.zh-CN.md)

# `experiments/visualize.py` — 出图命令行

`visualize.py` 读取一个或多个运行目录，输出 PNG 图表。用
`uv run --group experiments` 运行。

## 选项

### `--dir DIR`（可重复，至少给一次）
- **作用**：运行目录，支持以下两种形式（自动判断）：
  - run-name 层，如 `examples/runs/TBLS Full`——自动选其下最新的
    `YYYYMMDD_HHMMSS` 子目录；
  - run-name/timestamp 层，如
    `examples/runs/TBLS Full/20260724_074140`——直接使用。
- **冲突**：更深（`.../<timestamp>/logs`）、更浅（`examples/runs`）、
  或子目录名不是 `YYYYMMDD_HHMMSS` 格式的路径都会**报错并给出明确提示**
  ——不需要 shell 通配。
- **空格**：run name 带空格没问题，shell 上加引号即可
  （`--dir "examples/runs/TBLS Full"`），图例中的空格会保留。
- **多个运行**：`--dir X --dir Y --dir Z` 把所有运行叠在同一张图上。

### `--output-dir DIR`
- **作用**：PNG 输出目录。
- **默认**：第一个 `--dir` 旁边的 `plots/`。

### `--dpi N`
- **作用**：PNG 分辨率。
- **默认**：`300`（印刷级）。快速预览可用 `--dpi 120`。

## 输出的图

在 `--output-dir` 下：

| 文件 | 何时生成 | 内容 |
|---|---|---|
| `per_fold_metrics.png` | 始终 | 各 cohort 的 `balanced_accuracy` 和 `mcc` 柱状图，按运行分组——核心消融对比图 |
| `grid_search_summary.png` | 仅当至少一个 `--dir` 是网格搜索运行 | 指标 vs. 每个被扫轴，每轴一个子图 |
| `roc_<cohort>.png` | 仅非网格运行 | 每个 cohort 一个 ROC 文件（如 `roc_DM.png`、`roc_CKD.png`），同一 cohort 的所有运行叠在同一坐标系上 |
| `pr_<cohort>.png` | 仅非网格运行 | 同上，precision-recall 曲线 |
| `confusion_<run>.png` | 仅非网格运行 | 每个运行一张混淆矩阵图（cohort 作子图） |

如果 `--dir` 列表中有网格搜索运行，它会正常报各折指标，但**跳过**
ROC/PR/混淆图（网格搜索不写 `.npz` 预测文件），并在终端打印提示。

## 为什么 ROC/PR 按 cohort 拆分

消融实验要回答的问题是"运行 A 在 cohort X 上是否胜过运行 B？"，而不是
"运行 A 的曲线跨 cohort 比如何"。所以每个 `roc_<cohort>.png` 把所有运行
叠在同一坐标系，每个 cohort 一个文件。数据来源见 [outputs.md](outputs.md)。

## 为什么 TBLS 的 PR 曲线会有陡峭悬崖

TBLS 的 `predict_proba` 是 ridge 闭式输出经 softmax 变换，**没有概率
标定步骤**，所以低置信度样本会集中在 `p≈0.5` 附近形成高密度带。阈值扫描
一步跨过这个带，把一大块多为负的样本同时标为"预测正"——精确率骤降到
流行率，召回率跳升。完整数学推导和复现器见
[figures-and-calibration.md](figures-and-calibration.md)。

## 常用命令

```bash
# 消融对比：所有运行叠在同一张图
uv run --group experiments python experiments/visualize.py \
    --dir "examples/runs/TBLS" \
    --dir "examples/runs/TBLS Full" \
    --dir "examples/runs/Logistic Regression" \
    --output-dir examples/plots --dpi 300

# 单个运行（看标量指标柱状图 + ROC/PR/混淆矩阵）
uv run --group experiments python experiments/visualize.py \
    --dir "examples/runs/TBLS Full"
```

## 常见报错

- **`--dir ... has no YYYYMMDD_HHMMSS timestamp subdirectory`** — 给了
  run-name 层，但下面没有 timestamp 子目录（运行可能失败了）。用
  `ls examples/runs/{run_name}` 检查。
- **`--dir ... has no 'logs/' subdirectory`** — 给的路径是 timestamp
  目录，但里面没有 `logs/`（可能是 cohort Excel 目录）。请给持有
  `logs/` 的 run-name 层或 run-name/timestamp 层。
