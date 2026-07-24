[English](./outputs.md) | 简体中文

# 运行后磁盘上有什么

这是 `train.py` 写出的每个文件的参考:Excel sheet 布局、JSONL 事件
schema、`.npz` side-file schema。

## 目录布局

`train.py` 写(相对 `output_dir`,默认 `results_dir/`):

```
{output_dir}/{run_name}/{timestamp}/                       <- 运行目录
    logs/{dataset}_{timestamp}.jsonl                       <- 结构化日志
    logs/{dataset}_{timestamp}_{cohort}_predictions.npz    <- 原始预测
                                                             (仅非网格运行)

{output_dir}/{run_name}/{cohort}/{timestamp}/              <- cohort 目录
    {cohort}_{model_name}_FS-{fs}_RS-{rs}.xlsx             <- 各 cohort Excel
```

`{timestamp}` 是 `time.strftime("%Y%m%d_%H%M%S")`,两条分支共用(运行目录
和 cohort 目录是 `{run_name}` 下的 timestamp 兄弟)。`{run_name}` 既作
目录段名也作 YAML 标签;`examples/runs/`/`plots/`/`examples/comparison/`
都被 git 忽略。

## Excel(`{cohort}_{model_name}_FS-..._RS-..._xlsx`)

每个 cohort 一个文件。`TBLSResultSaver` 随运行进展累加 sheet。

### 非网格运行

| Sheet | 行 |
|---|---|
| `{model}_Details` | 每 CV 折一行,带每个折级指标 |
| `{model}_Summary` | 一行,跨折平均(键前缀 `avg_`)+ `{key: cohort}` |
| `{sheet}_Meta` | 一行,`Feature_Selection`/`Resampling_Method` 元数据 |

### `--grid` 运行

| Sheet | 行 |
|---|---|
| 每网格点 `Grid_{i:03d}` | 第 `i` 个网格点的 `_cross_validate` 折级 dump |
| `GridSummary` | 每网格点一行,按 `avg_balanced_accuracy` 降序;被扫超参 + 每个 `avg_*` 指标 + `rank` + `is_winner`(**第 1 行 = 冠军**) |
| `GridSearchLog` | 同 `GridSummary` 但平铺(排名前的时间序) — 当搜索日志读 |
| `{sheet}_Meta` | 特征选择 + 重采样元数据 |

## JSONL 日志(`logs/{dataset}_{timestamp}.jsonl`)

运行开始时由 `experiments/logging_setup.py` 配两个 sink:人类可读 stdout
(INFO,彩色)和这个 JSONL 文件(DEBUG,`serialize=True`)。每行一条记录:
loguru 元数据在 `record.{text, level, time, function, line, ...}`,任何
bound 的事件载荷在 `record.extra`。

事件(`event` 是 `record.extra` 里的判别字段):

### `run_started`(每运行一次)
| 字段 | 类型 | 备注 |
|---|---|---|
| `dataset` | str | 数据集名 |
| `model` | str | 模型名(`"tbls"`、`"bls"` 或基线名) |
| `fusion_method` | str \| None | 仅多视图 cohort;否则 `None` |
| `grid` | bool | 是否传了 `--grid` |
| `run_name` | str \| None | 仅 YAML `run_name:` 设了才有;否则缺省 |

### `fold_completed`(每折每 cohort)
| 字段 | 类型 | 备注 |
|---|---|---|
| `dataset`,`cohort_key` | str | — |
| `fold`,`n_splits` | int | 1 起折号;总折数 |
| `metrics` | `MetricsDict` | 每个折级标量指标(见下"指标键") |
| `grid_idx` | int \| None | 1 起网格点号(仅 `--grid`;否则 `None`) |
| `grid_params` | dict \| None | 该点被扫超参(仅 `--grid`;否则 `None`) |
| `predictions_file` | str \| None | `.npz` side-file 名(仅非网格折;否则 `None`) |

### `grid_point_completed`(每被扫网格点,`--grid` 下)
| 字段 | 类型 | 备注 |
|---|---|---|
| `dataset`,`cohort_key` | str | — |
| `grid_idx`,`n_grid_points` | int | 1 起;总点数 |
| `grid_params` | dict | 该点超参 |
| `metrics` | dict | 跨折平均指标(前缀 `avg_`) |

### `grid_summary`(每 cohort 一次,`--grid` 排名后)
| 字段 | 类型 | 备注 |
|---|---|---|
| `dataset`,`cohort_key` | str | — |
| `winner_params` | dict | 冠军超参行 |
| `winner_metric` | float | 冠军的 `avg_balanced_accuracy` |
| `n_grid_points` | int | 总扫点数 |

### `run_finished`(每运行一次)
| 字段 | 类型 | 备注 |
|---|---|---|
| `dataset` | str | — |
| `duration_seconds` | float | 挂钟运行时长 |

标准库 `logging` 也被拦截(`InterceptHandler` 把
`logging.getLogger(...).warning(...)` 转发进同一 JSONL),所以例如
`experiments.evaluate` 的概率指标警告也出现在这里。

### `fold_completed.metrics` 里的指标键

schema 是 `experiments/metrics_schema.py` 的 `MetricsDict` TypedDict。
二分类路径返回:`accuracy`、`precision`、`recall`、`f1_score`、
`hamming_loss`、`specificity`、`negative_predictive_value`、
`balanced_accuracy`、`gmean`、`mcc`、`cohen_kappa`,以及(有 `y_score`
时)`auroc`、`auprc`、`optimal_threshold`、`log_loss`、`brier_score`。
多分类路径返回相同键,但二分类专属键(`auprc`/`optimal_threshold`/
`log_loss`/`brier_score`)缺省;另外加 `precision_weighted`/
`recall_weighted`/`f1_weighted`,并直接用 `balanced_accuracy_score`。

概率派生键(`auroc`/...)在 `y_score` 缺失或底层 sklearn 在退化折抛错时
降级为 `None`。

## `.npz` 预测 side-file

非网格折会在 `logs/` 下写一个 per-cohort `.npz`,命名
`{dataset}_{timestamp}_{cohort}_predictions.npz`。每文件按折存:

| 键 | dtype/形状 |
|---|---|
| `{cohort}_fold{N}_y_true` | `int64` 形 `(n_test,)` |
| `{cohort}_fold{N}_y_pred` | `int64` 形 `(n_test,)` — `model.predict(X_te)` |
| `{cohort}_fold{N}_y_score` | `float32` 形 `(n_test, n_classes)` — `model.predict_proba(X_te)` |

`y_score` 是 2-D 概率矩阵,**不是** 1-D 正类向量。要二分类正类的读者
须切片 `y_score[:, 1]` — `visualize.py` 和 `compare.py` 已这么做。

网格运行**不**写 side-file(27 × n_folds × n_cohorts 会爆文件)。因此网格
运行的 `fold_completed.predictions_file` 是 `None`。
