English | [简体中文](./grid-search.zh-CN.md)

# 超参网格搜索

`--grid` 把 `train.py` 从单次 k-fold CV 切换为超参搜索。本页说明扫什么、
怎么定制、输出什么。

## 一句话总结

| 设置 | `--grid` 扫什么 |
|---|---|
| `model.name: tbls`，无 YAML `grid:` | 默认 `TBLS_GRID` = `n_map_trees: [10,20,40] × n_enhance_trees: [10,20,40] × reg_param: [1e-8,1e-4,1e-2]` = 27 个点 |
| `model.name: bls`，无 YAML `grid:` | `BLS_GRID` = `n_feature_groups: [15,30,60] × n_feature_nodes_per_group: [20,40,80] × reg_param: [0.1,1.0,10.0]` = 27 个点 |
| `model.name: tbls|bls`，有 YAML `grid:` | 默认 ∪ YAML，YAML 在轴名冲突时优先 |
| `model.name: <基线>`，有 YAML `grid:` | 仅 YAML（基线**没有**默认网格） |
| `model.name: <基线>`，无 YAML `grid:` | **`--grid` 被忽略**：告警 + 退化为单次 k-fold |

## 怎么用

给 `train.py` 加 `--grid`：

```bash
uv run --group experiments python experiments/train.py \
    --config examples/configs/default.yaml --grid
```

要扫别的参数，在 YAML 里加 `grid:` 块。两个例子：

**例 A——只扫 IFS + 图两个轴，其余钉死：**
```yaml
model:
  name: tbls
  use_if_weights: false
  graph_gamma: 0.0
grid:
  use_if_weights: [false, true]
  graph_gamma: [0.0, 0.05, 0.1]
```
2 × 3 = 6 个点。没在 `grid:` 里提到的默认轴保持各自的单个默认值，不会
被扫。

**例 B——扫一个没默认网格的基线：**
```yaml
model: {name: lr}
grid:
  C: [0.01, 0.1, 1.0, 10.0]
```
4 个点。没有这个 `grid:` 段的话，基线的 `--grid` 会退化为单次 k-fold
并打告警。

## 合并规则

`train.py::_resolve_grid` 的逻辑：

1. **取默认**——`tbls` 用 `TBLS_GRID`，`bls` 用 `BLS_GRID`，基线无。
2. **叠上 YAML `grid:`**——YAML 里出现的轴**替换**同名默认轴的值列表；
  没提到的默认轴**保留**。
3. 基线 + 无 YAML `grid:` → 报错/退化为单次 CV，绝不静默。

一个机制，三种用法：
- **收缩**：`grid: {reg_param: [1e-8, 1e-4]}`——只扫这一个轴，其余保持
  默认值。
- **替换**：`grid: {n_map_trees: [5, 50]}`——覆盖该轴的默认值列表。
- **新增**：`grid: {use_if_weights: [false, true]}`——加一条新轴（TBLS
  构造器接受它，会作为网格点来扫）。

## 输出

每个 `--grid` cohort 生成：

| 输出 | 内容 |
|---|---|
| Excel `Grid_{i:03d}` sheet | 第 `i` 个网格点的各折指标 |
| Excel `GridSummary` sheet | 每个网格点一行，按 `avg_balanced_accuracy` 降序排名，含 `rank` 和 `is_winner` 列——**第 1 行是冠军** |
| Excel `GridSearchLog` sheet | 同 `GridSummary` 但按时间序排列（未排名），当搜索日志读 |
| JSONL `grid_point_completed` 事件（每点一个） | `grid_idx`、`n_grid_points`、`grid_params`、平均指标 |
| JSONL `grid_summary` 事件（结尾） | `winner_params`、`winner_metric`、`n_grid_points` |

完整事件 schema 和 Excel 布局见 [outputs.md](outputs.md)。

把 `visualize.py` 指向网格搜索运行会产出 `grid_search_summary.png`：
主指标 vs. 每个被扫轴，每轴一个子图。详见
[cli-visualize.md](cli-visualize.md)。

## 范围限制

- **`--grid` 不扫融合超参**（`CCA_GRID`/`GFCCA_GRID`）。即使多视图 cohort
  也只扫模型网格。这是文档化的范围限制，不是静默丢弃。见
  [../usage-multiview-fusion.md](../usage-multiview-fusion.md)。
- **网格搜索不写 `.npz` 预测文件**（27 点 × N 折 × N cohort 体积太大）。
  因此网格搜索跳过 ROC/PR/混淆图；各折柱状图和网格搜索汇总图正常输出。
  详见 [cli-visualize.md](cli-visualize.md)。
