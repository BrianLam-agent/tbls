English | [简体中文](./internals.zh-CN.md)

# 流水线内部（给维护者）

本页给想**改或扩展流水线代码**的人看。光跑的话从 [index.md](index.md) 起。
本页假设你已读过 cli-*/outputs 各页，现在想知道"每块为什么这么设计"。

## 模块地图（`experiments/`）

| 文件 | 职责 |
|---|---|
| `train.py` | 编排器：typer CLI、YAML 加载、run/cohort/fold 循环、模型构建、评估、Excel/日志写入。顶部用脚本式兄弟导入（`from dataprocess import ...`），需要 `experiments/` 在 `sys.path`——`tests/conftest.py` 设了；CLI 经 `uv run experiments/train.py` 跑通是因为 `experiments/` 是脚本自身目录。 |
| `classifiers.py` | 基线工厂 `create_classifier(name, ...)`。每个支持的基线名一个小 `if name == ...:` 分支，构建带类别均衡默认的 sklearn 包装估计器。软依赖（`xgboost`/`lightgbm`/`catboost`/`torch`）懒加载，只在请求对应 `name` 时才抛 `ImportError`。 |
| `dataprocess.py` | `DataLoader`——pkl/CSV 加载器 + 特征选择 + 重采样。旧 CSV pair 路径（`_load_csv`）用 `MultiLabelBinarizer`；pkl 路径二值化到 `{0, 1}`。CSV 路径*故意*不是二分类 TBLS 流水线走的——它是旧多标签工作的钩子。 |
| `evaluate.py` | `TBLSEvaluator`（标量指标，二分类 + 多分类分发器）+ `TBLSResultSaver`（Excel 写入器）。 |
| `metrics_schema.py` | `MetricsDict` TypedDict——规范的折级指标 schema。`evaluate.py` 与 `logging_schema.py` 共享以避免循环导入。 |
| `hyperparams.py` | 模块级 Python dict 常量 `TBLS_DEFAULTS`/`BLS_DEFAULTS`/`TBLS_GRID`/`BLS_GRID`/`CCA_*`/`GFCCA_*`。不是 YAML（故意的：值受版本控制 + 代码评审）。 |
| `logging_setup.py` | `configure_logging(output_dir, dataset, timestamp)`：双 loguru sink（stdout INFO + JSONL DEBUG `serialize=True`）+ 标准库 `InterceptHandler`。 |
| `logging_schema.py` | 5 个事件 TypedDict：`RunStartedEvent`、`FoldCompletedEvent`、`GridPointCompletedEvent`、`GridSummaryEvent`、`RunFinishedEvent`。 |
| `multiview.py` | `MultiViewDataLoader` + `load_multiview_cohort` + `fuse_views`。单视图 `_cross_validate` 路径完全绕过它。 |
| `run_resolution.py` | `resolve_run_dir(run_arg)` / `cohort_excel_dir(run_dir, cohort)`——规范的 `--dir` 解析规则（run-name 层 vs run-name/timestamp vs 畸形路径）。`visualize.py` 与 `compare.py` 共享。 |
| `visualize.py` | `/visualize` CLI：解析 JSONL 事件 + npz 预测文件，渲染 matplotlib PNG。 |
| `compare.py` | `/compare` CLI：跨运行解析 JSONL fold 事件，写 `comparison.xlsx`，单元格 `mean (std)`，按 `METRIC_DIRECTION` 给每个 (cohort, 指标) 最优值加粗。 |
| `smoke_run.py` | 单 k-split 健全性检查 + 一个被示例脚本和真实数据集测试复用的 `_extract_xy` 助手。 |

## `model.name` 在哪分发

`train.py::_build_model(model_cfg, grid_point=None)`：

```python
if name == "tbls":  defaults = TBLS_DEFAULTS    ; cls = TBLS
elif name == "bls": defaults = BLS_DEFAULTS     ; cls = BroadLearningSystem
else:  return create_classifier(name, random_state=..., **YAML_kwargs)
```

对 `tbls`/`bls`，YAML 键按构造器签名过滤（旧 `map_num`/`enhance_num` →
`n_map_trees`/`n_enhance_trees`），然后 `grid_point` 覆盖优先。对基线，
YAML 键直接以 `**kwargs` 传给 `create_classifier`（不过滤签名——传了该
sklearn 估计器不接受的 kwarg，构造器会抛），且 `grid_point` **被忽略**
（基线不参与 `--grid`）。

## 两层网格解析（`_resolve_grid`）

- **默认**：`tbls` 用 `TBLS_GRID`，`bls` 用 `BLS_GRID`，基线无。
- **YAML `grid:`**：若 cfg 有 `grid:`，解析网格是默认的拷贝，然后 YAML
  里每个命名轴**替换**同名轴的值列表。只在 YAML 不在默认里的轴被新增
  （所以扫基线就只给 YAML `grid:`）。
- 无 YAML `grid:` 的基线在 `_resolve_grid` *内部*抛
  `ValueError("No default grid for ...")`——但 `train.py` 调用方先检查，
  对裸 `--grid` 基线退化为单 k-fold + 告警而不是去调 `_resolve_grid`，
  所以用户路径是优雅的。

## `--dir` 解析（`run_resolution.resolve_run_dir`）

规则机械执行：

1. 若 `run_arg.name` 匹配 `^\d{8}_\d{6}$` → 当 timestamp 层直接用。验证
  有 `logs/` 子目录；验证无 timestamp-下-timestamp 嵌套（太深）。
2. 否则 → 搜 `run_arg/*` 找 `YYYYMMDD_HHMMSS` 子目录，取字典序最大
  （最新），验证其 `logs/` 子目录存在。
3. 其他任何形态（比 run-name 浅、比 `<ts>/logs` 深、或 timestamp 目录
  的父目录不是 run-name 层）都抛带类型异常，附一行诊断。

`cohort_excel_dir(run_dir, cohort)` 返回
`run_dir.parent / cohort / run_dir.name`（同 timestamp 的兄弟 cohort
目录），路径不存在则抛——所以 `compare.py` 不会静默拉一个不匹配 timestamp
的 cohort 目录。

## 原始预测持久化（`_cross_validate`）

仅非网格折把每折 `y_true`/`y_pred`/`y_score` 累进
`preds[{cohort}_fold{N}_{y_true,y_pred,y_score}]`，在 `_cross_validate`
结尾 `np.savez`。网格运行传 `predictions_npz=None`，所以 npz 跳过——
体积原因见 git log 里 Plan 02/06 记录（~c811812 区域）。`fold_completed`
事件的 `predictions_file` 字段带 npz *名*（无路径），消费者相对 JSONL
自身 `logs/` 目录加载。

## 没在发生的事（省得你去找）

- 没有概率标定。`TBLS.predict_proba` 是闭式 ridge 输出经 softmax；产生
  PR 悬崖的 `0.5` 得分密度平台是模型伪影，不是流水线伪影。见
  [figures-and-calibration.md](figures-and-calibration.md)。
- `--grid` 不扫 CCA/GFCCA 融合轴。（文档化的范围限制。）
- `DataLoader` 里的旧 CSV 路径不与二分类流水线交互；别试图让它交互。
- `run_resolution` 故意不接受含多个运行的裸目录（`examples/runs`）——
  每个运行必须单独寻址。批处理是 `train.py` 层的 `--config-dir`，不是
  run-resolution 层的。
