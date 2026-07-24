English | [简体中文](./cli-compare.zh-CN.md)

# `experiments/compare.py` — 跨运行对比 Excel

`compare.py` 读取一个或多个运行目录（`--dir` 规则与 `visualize.py` 相同），
输出一个 `comparison.xlsx`，以论文表格形式汇总各运行各 cohort 的指标。
用 `uv run --group experiments` 运行。

## 选项

### `--dir DIR`（可重复，至少给一次）
- **作用**：运行目录，支持以下两种形式（自动判断）：
  - run-name 层——自动选最新 `YYYYMMDD_HHMMSS` 子目录；
  - run-name/timestamp 层——直接使用。
- **冲突**：更深/更浅/非 timestamp 路径会报错，规则同
  [cli-visualize.md](cli-visualize.md)。
- **重复标签**：两个 `--dir` 解析到同一个运行名会抛
  `ValueError: Duplicate run label ...`。

### `--output-dir DIR`
- **作用**：`comparison.xlsx` 输出目录。
- **默认**：`examples/comparison`。

### `--no-std`
- **作用**：去掉单元格中的 `(std)`，只写裸均值。
- **默认**：关闭（每个单元格为 `mean (std)`）。

## 输出：`comparison.xlsx`

每个 cohort 一个 Excel sheet，加一个 `README` sheet：

| Sheet | 内容 |
|---|---|
| `README` | 布局说明 + 各指标方向表（越高越好还是越低越好） |
| `<cohort>` | 行 = 运行（已排序），列 = 15 个标量指标 |

每个单元格为跨 CV 折的 `mean (std)`，如 `0.9237 (0.0112)`。加 `--no-std`
后变为裸均值如 `0.9237`。

### 15 个指标及方向（决定加粗）

`compare.py` 对每个 (cohort, 指标) **加粗最优运行**。"最优"取决于该指标
是越高越好还是越低越好：

| 指标 | 方向（加粗 ... 的运行） |
|---|---|
| `balanced_accuracy` | 均值最高 |
| `accuracy` | 最高 |
| `f1_score` | 最高 |
| `mcc` | 最高 |
| `cohen_kappa` | 最高 |
| `auroc` | 最高 |
| `auprc` | 最高 |
| `recall` | 最高 |
| `specificity` | 最高 |
| `precision` | 最高 |
| `negative_predictive_value` | 最高 |
| `gmean` | 最高 |
| `hamming_loss` | 最低（越低越好） |
| `log_loss` | 最低 |
| `brier_score` | 最低 |

某个运行没产出某个 cohort 时，该单元格留空（不是 0、不是 NaN），一眼
就能看出缺了哪个 cohort。

## 数据来源

`compare.py` 从每个运行的 `logs/{dataset}_{timestamp}.jsonl` 中解析
`fold_completed` 事件（不是从 Excel 文件），计算各指标跨折的均值和标准
差。指标 schema 见 [outputs.md](outputs.md)。

## 常用命令

```bash
# 5 个运行消融对比：均值 ± 标准差，每指标最优加粗
uv run --group experiments python experiments/compare.py \
    --dir "examples/runs/TBLS" \
    --dir "examples/runs/TBLS IFS" \
    --dir "examples/runs/TBLS Graph" \
    --dir "examples/runs/TBLS Full" \
    --dir "examples/runs/Logistic Regression" \
    --output-dir examples/comparison

# 只要裸均值（电子表格里更顺手）：加 --no-std
uv run --group experiments python experiments/compare.py \
    --dir "examples/runs/TBLS" \
    --dir "examples/runs/Logistic Regression" --no-std
```

## 常见报错

- **`--dir ... has no YYYYMMDD_HHMMSS timestamp subdirectory`** — 同
  visualize.py：给了 run-name 层但下面没有 timestamp 子目录。
- **`Per-cohort Excel dir not found: ...`** — `compare.py` 找与 JSONL
  运行目录同 timestamp 的兄弟 cohort Excel 目录；如果运行后移动过文件，
  可能找不到了。重跑 `train.py` 重建，或换一个 timestamp。
