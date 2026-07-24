[English](./cli-visualize.md) | 简体中文

# `experiments/visualize.py` — 出图 CLI

`visualize.py` 读一个或多个运行目录,输出 PNG 图表。用
`uv run --group experiments` 运行。

## 选项

### `--dir DIR`(可重复;至少一次)
- **干啥**:一个运行目录,可以是下面任一形态(CLI 自动判别):
  - run-name 层,如 `examples/runs/TBLS Full` — CLI 自动选其下最新的
    `YYYYMMDD_HHMMSS` timestamp 子目录;
  - run-name/timestamp 层,如
    `examples/runs/TBLS Full/20260724_074140` — 直接用。
- **冲突**:更深(`.../<timestamp>/logs`)、更浅(`examples/runs`)、或其
  子目录不是 `YYYYMMDD_HHMMSS` 名字的路径**都会以清晰诊断报错** — 不需要
  shell 通配。
- **空格**:run name 带空格没问题 — shell 上把路径加引号
  (`--dir "examples/runs/TBLS Full"`);图例里的空格会保留。
- **多次运行**:传 `--dir X --dir Y --dir Z` 即可在同一张图上叠加所有运行。

### `--output-dir DIR`
- **干啥**:PNG 写到哪。
- **默认**:第一个 `--dir` 旁的 `plots/`。

### `--dpi N`
- **干啥**:PNG 分辨率。
- **默认**:`300`(印刷级)。预览快图用 `--dpi 120`。

## 产出的图

在 `--output-dir` 下:

| 文件 | 何时产生 | 内容 |
|---|---|---|
| `per_fold_metrics.png` | 总是 | 各 cohort 的 `balanced_accuracy` 和 `mcc` 柱状图,按运行分组 — 核心消融对比图。 |
| `grid_search_summary.png` | 仅当至少一个 `--dir` 是 `--grid` 运行 | 指标 vs. 每个被扫轴,每轴一个子图。 |
| `roc_<cohort>.png` | 仅非网格运行 | 每个 cohort 一个 ROC 文件(`roc_DM.png`、`roc_CKD.png`、...)。每个文件把所有运行的 ROC 叠加在**该 cohort** 上 — 这才是消融对比该有的样子。 |
| `pr_<cohort>.png` | 仅非网格运行 | 同上模式,precision-recall 曲线。 |
| `confusion_<run>.png` | 仅非网格运行 | 每运行一张混淆矩阵图(cohort 作子图)。 |

若 `--dir` 列表里有网格运行,它会照常报各折指标,但**跳过** ROC/PR/混淆
图(网格运行不写 `.npz` 预测 side-file),并在 stdout 打个提示 — 同样
行为。

## 为啥 ROC/PR 按 cohort 拆

消融要回答的自然问题是"运行 A 在 cohort X 上是否胜过运行 B?",而不是
"运行 A 的曲线跨 cohort 比如何"。所以每个 `roc_<cohort>.png` 把所有运行
叠在同一坐标系,每个 cohort 一个文件,而不是一张混杂 cohort 图例。喂给
曲线的 npz 布局见 [outputs.md](outputs.md)。

## 为啥 TBLS 的 PR 曲线会有陡悬崖

TBLS 的 `predict_proba` 是 ridge 闭式输出经 softmax 变换,**没有概率标定
步骤**,所以低置信度样本会形成一个 `p≈0.5` 的高密度带。阈值扫描一步跨过
这带,把一大块多为负样本同时塞进"预测为正" → 精确率崩到流行率、召回率
跳升。完整数学 + 复现器:
[figures-and-calibration.md](figures-and-calibration.md)。

## 典型调用

```bash
# 一站式消融对比:把所有示例运行叠在同一张图
uv run --group experiments python experiments/visualize.py \
    --dir "examples/runs/TBLS" \
    --dir "examples/runs/TBLS Full" \
    --dir "examples/runs/Logistic Regression" \
    --output-dir examples/plots --dpi 300

# 只一次运行(看标量指标柱状图 + npz 派生图)
uv run --group experiments python experiments/visualize.py \
    --dir "examples/runs/TBLS Full"
```

## 常见错误

- **`--dir ... has no YYYYMMDD_HHMMSS timestamp subdirectory`** — 你给了
  run-name 层,但其下没有 timestamp 子目录(运行可能失败了)。跑
  `ls examples/runs/{your run_name}`。
- **`--dir ... has no 'logs/' subdirectory; not a valid run timestep
  directory`** — 你给的路径*是*一个 `YYYYMMDD_HHMMSS` 目录,但里面不是
  `logs/` 运行(可能是各 cohort Excel 目录之类)。请给持有 `logs/` 的
  run-name 层或 run-name/timestamp 层。
