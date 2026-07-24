English | [简体中文](./outputs.zh-CN.md)

# 运行结束后磁盘上有什么

本文档说明 `train.py` 写出的每一个文件：目录布局、Excel sheet 内容、
JSONL 事件 schema、`.npz` 预测文件格式。

## 目录布局

`train.py` 在 `output_dir`（默认 `results_dir/`）下生成：

```
{output_dir}/{run_name}/{timestamp}/                       ← 运行目录
    logs/{dataset}_{timestamp}.jsonl                       ← 结构化日志
    logs/{dataset}_{timestamp}_{cohort}_predictions.npz    ← 原始预测
                                                             （仅非网格运行）

{output_dir}/{run_name}/{cohort}/{timestamp}/              ← cohort 目录
    {cohort}_{model_name}_FS-{fs}_RS-{rs}.xlsx             ← 各 cohort Excel
```

`{timestamp}` 是 `time.strftime("%Y%m%d_%H%M%S")`，运行目录和 cohort
目录共用同一个 timestamp（两者是 `{run_name}` 下的兄弟目录）。`{run_name}`
同时作为目录段名和 YAML 标签；`examples/runs/`、`plots/`、
`examples/comparison/` 都已 gitignore。

## Excel（`{cohort}_{model_name}_FS-..._RS-..._xlsx`）

每个 cohort 一个文件。`TBLSResultSaver` 随运行进展逐个添加 sheet。

### 非网格运行

| Sheet | 内容 |
|---|---|
| `{model}_Details` | 每 CV 折一行，含所有折级指标 |
| `{model}_Summary` | 一行，跨折平均值（键前缀 `avg_`）+ cohort 键 |
| `{sheet}_Meta` | 一行，`Feature_Selection`/`Resampling_Method` 元数据 |

### 网格搜索运行

| Sheet | 内容 |
|---|---|
| 每个网格点 `Grid_{i:03d}` | 第 `i` 个网格点的各折指标 |
| `GridSummary` | 每个网格点一行，按 `avg_balanced_accuracy` 降序排名，含被扫超参、所有 `avg_*` 指标、`rank` 和 `is_winner`——**第 1 行是冠军** |
| `GridSearchLog` | 同 `GridSummary` 但按时间序排列（未排名），当搜索日志读 |
| `{sheet}_Meta` | 特征选择 + 重采样元数据 |

## JSONL 日志（`logs/{dataset}_{timestamp}.jsonl`）

运行开始时由 `experiments/logging_setup.py` 配置两个 sink：人类可读的
stdout（INFO，彩色）和这个 JSONL 文件（DEBUG，`serialize=True`）。每行
一条记录：loguru 元数据在 `record.{text, level, time, function, line, ...}`，
事件载荷在 `record.extra`。

事件以 `event` 字段区分：

### `run_started`（每运行一次）
| 字段 | 类型 | 说明 |
|---|---|---|
| `dataset` | str | 数据集名 |
| `model` | str | 模型名（`"tbls"`、`"bls"` 或基线名） |
| `fusion_method` | str \| None | 仅多视图 cohort 有值，否则 `None` |
| `grid` | bool | 是否传了 `--grid` |
| `run_name` | str \| None | 仅 YAML `run_name:` 设了才有 |

### `fold_completed`（每折每 cohort）
| 字段 | 类型 | 说明 |
|---|---|---|
| `dataset`, `cohort_key` | str | — |
| `fold`, `n_splits` | int | 折号从 1 起；总折数 |
| `metrics` | `MetricsDict` | 所有折级标量指标（见下方"指标键"） |
| `grid_idx` | int \| None | 网格点号（仅 `--grid`，从 1 起；否则 `None`） |
| `grid_params` | dict \| None | 该点被扫超参（仅 `--grid`；否则 `None`） |
| `predictions_file` | str \| None | `.npz` 文件名（仅非网格折；否则 `None`） |

### `grid_point_completed`（每网格点，仅 `--grid`）
| 字段 | 类型 | 说明 |
|---|---|---|
| `dataset`, `cohort_key` | str | — |
| `grid_idx`, `n_grid_points` | int | 网格点号（从 1 起）；总点数 |
| `grid_params` | dict | 该点超参 |
| `metrics` | dict | 跨折平均指标（前缀 `avg_`） |

### `grid_summary`（每 cohort 一次，`--grid` 排名后）
| 字段 | 类型 | 说明 |
|---|---|---|
| `dataset`, `cohort_key` | str | — |
| `winner_params` | dict | 冠军超参 |
| `winner_metric` | float | 冠军的 `avg_balanced_accuracy` |
| `n_grid_points` | int | 总网格点数 |

### `run_finished`（每运行一次）
| 字段 | 类型 | 说明 |
|---|---|---|
| `dataset` | str | — |
| `duration_seconds` | float | 挂钟运行时长 |

标准库 `logging` 也被拦截（`InterceptHandler` 把
`logging.getLogger(...).warning(...)` 转发进同一个 JSONL），所以
`experiments.evaluate` 的概率指标警告也会出现在这里。

### `fold_completed.metrics` 中的指标键

schema 定义在 `experiments/metrics_schema.py` 的 `MetricsDict` TypedDict
中。二分类路径返回：`accuracy`、`precision`、`recall`、`f1_score`、
`hamming_loss`、`specificity`、`negative_predictive_value`、
`balanced_accuracy`、`gmean`、`mcc`、`cohen_kappa`，以及（有 `y_score`
时）`auroc`、`auprc`、`optimal_threshold`、`log_loss`、`brier_score`。
多分类路径返回相同键，但 `auprc`/`optimal_threshold`/`log_loss`/
`brier_score` 缺省，另外加 `precision_weighted`/`recall_weighted`/
`f1_weighted`。

`y_score` 缺失或底层 sklearn 在退化折上抛错时，概率相关指标降级为
`None`。

## `.npz` 预测文件

非网格折会在 `logs/` 下写一个 per-cohort `.npz`，文件名为
`{dataset}_{timestamp}_{cohort}_predictions.npz`。每折存储：

| 键 | dtype / 形状 |
|---|---|
| `{cohort}_fold{N}_y_true` | `int64`，形状 `(n_test,)` |
| `{cohort}_fold{N}_y_pred` | `int64`，形状 `(n_test,)`——`model.predict(X_te)` |
| `{cohort}_fold{N}_y_score` | `float32`，形状 `(n_test, n_classes)`——`model.predict_proba(X_te)` |

`y_score` 是 2-D 概率矩阵，**不是** 1-D 正类向量。二分类取正类概率需
切片 `y_score[:, 1]`——`visualize.py` 和 `compare.py` 已做此处理。

网格搜索**不**写 `.npz`（27 × N 折 × N cohort 体积太大），因此网格
运行的 `fold_completed.predictions_file` 为 `None`。
