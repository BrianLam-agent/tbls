[English](./grid-search.md) | 简体中文

# 超参网格搜索

`--grid` 把 `train.py` 从单次 k-fold CV 切到超参扫描。本页告诉你扫什么、
怎么定制、磁盘上落什么。

## TL;DR

| 设置 | `--grid` 扫什么 |
|---|---|
| `model.name: tbls`,无 YAML `grid:` | 默认 `TBLS_GRID` = `n_map_trees: [10,20,40] × n_enhance_trees: [10,20,40] × reg_param: [1e-8,1e-4,1e-2]` = 27 点 |
| `model.name: bls`,无 YAML `grid:` | `BLS_GRID` = `n_feature_groups: [15,30,60] × n_feature_nodes_per_group: [20,40,80] × reg_param: [0.1,1.0,10.0]` = 27 点 |
| `model.name: tbls|bls`,有 YAML `grid:` | 默认 ∪ YAML,YAML 在轴名冲突时胜 |
| `model.name: <基线>`,有 YAML `grid:` | 仅 YAML(基线**没有**默认网格) |
| `model.name: <基线>`,无 YAML `grid:` | **`--grid` 被忽略**:告警 + 退化为单次 k-fold 运行 |

## 怎么用

给 `train.py` 调用加 `--grid`:

```bash
uv run --group experiments python experiments/train.py \
    --config examples/configs/default.yaml --grid
```

要扫别的东西,就在 YAML 里加 `grid:` 块。两个例子:

**例 A — 只扫 IFS + 图两个轴,其余钉死:**
```yaml
model:
  name: tbls
  use_if_weights: false   # 默认;每个网格点覆盖
  graph_gamma: 0.0        # 默认;每个网格点覆盖
grid:
  use_if_weights: [false, true]
  graph_gamma: [0.0, 0.05, 0.1]
# n_map_trees / n_enhance_trees / reg_param 没提 -> 也保持默认
```
这是 2 轴、6 点扫描(2 × 3) — 你没提到的 `TBLS_GRID` 命名轴不会扫(因为
它们都不在你 `grid:` 里)。但要紧的是:一旦你给了 `grid:`,每个你没点名的
默认轴会在扫描期间**保持**其全部默认值。所以例 A 实际只扫
`use_if_weights × graph_gamma`(6 点),其余 `TBLS` 超参用 YAML `model.`
默认。

**例 B — 扫一个没默认网格的基线:**
```yaml
model:
  name: lr
grid:
  C: [0.01, 0.1, 1.0, 10.0]
```
这是 LR 的 `C` 上 4 点扫描。没有这个 YAML `grid:`,基线的 `--grid` 会
静默退化为单次 k-fold 运行(并告警)。

## 解析规则(合并语义)

`train.py::_resolve_grid`:

1. **从默认开始** — `tbls` 用 `TBLS_GRID`,`bls` 用 `BLS_GRID`,基线无。
2. **把 YAML `grid:` 叠上去**:YAML 里每个命名轴会*替换*同名默认轴的值
   列表。你没提的默认轴*保留*。
3. 基线 + 无 YAML `grid:` → 明确报错/退单 CV,绝不静默。

一个机制支撑三种常见 move:
- **收缩**:`grid: {reg_param: [1e-8, 1e-4]}` — 只扫这轴,其余用各自
  (单默认值)值。
- **替换**:`grid: {n_map_trees: [5, 50]}` — 覆盖该轴默认列表。
- **新增**:`grid: {use_if_weights: [false, true]}` — 加一条全新轴
  (TBLS 构造器接受它作合法 kwarg;作为网格点扫)。

## 磁盘上落什么

每个 `--grid` cohort:

| 输出 | 内容 |
|---|---|
| Excel `Grid_{i:03d}` sheet | 第 `i` 个网格点的 `_cross_validate` 各折行 |
| Excel `GridSummary` sheet | 每网格点一行,按 `avg_balanced_accuracy` 降序排名;含 `rank` 和 `is_winner` 列 — **第 1 行是冠军** |
| Excel `GridSearchLog` sheet | 同 `GridSummary` 内容但平铺(按时间序,未排名) |
| JSONL `grid_point_completed` 事件(每点一个) | `grid_idx`、`n_grid_points`、`grid_params`、平均指标 |
| JSONL `grid_summary` 事件(结尾) | `winner_params`、`winner_metric`、`n_grid_points` |

(完整事件 schema 与 Excel 布局见 [outputs.md](outputs.md)。)

若把 `visualize.py` 指向 `--grid` 运行,会产出 **`grid_search_summary.png`**:
主指标 vs. 每个被扫轴,每轴一个子图(见 [cli-visualize.md](cli-visualize.md))。

## 范围限制(取自 Plan 规格,仍生效)

- **`--grid` 不扫融合超参**(`CCA_GRID` / `GFCCA_GRID`)。即便多视图 cohort,
  也只扫模型网格。这是文档化的范围限制 — 不是静默丢弃。见
  [../usage-multiview-fusion.md](../usage-multiview-fusion.md)。
- **原始 `.npz` 预测 side-file 对 `--grid` 运行不写**(体积 — 27 点 × N 折
  × N cohort 会爆)。`--grid` 运行跳过 ROC/PR/混淆图;各折柱状图和网格搜索
  汇总图照常。见 [cli-visualize.md](cli-visualize.md)。
