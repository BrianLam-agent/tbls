English | [简体中文](./config-reference.zh-CN.md)

# YAML 配置参考

YAML 配置驱动一次 `train.py` 运行。本页逐条列出每个键的含义、取值、默认
值和示例。*哪些 `model.name` 合法*见 [models.md](models.md)；*哪些命令行
选项可覆盖这些键*见 [cli-train.md](cli-train.md)。

## 顶层键

| 键 | 必填 | 类型 | 默认值 | 说明 |
|---|---|---|---|---|
| `dataset` | 是 | 字符串 | — | 数据集名，加载 `{data_path}/{dataset}.pkl` |
| `data_path` | 否 | 字符串 | `experiments/datasets` | pkl 所在目录 |
| `run_name` | 否 | 字符串 | `{model.name}_{dataset}` | 同时作为运行目录名和图表图例标签。写好以得清晰标签；允许空格 |
| `model` | 是 | mapping | `{name: tbls}` | 选模型及其构造器参数 |
| `preprocess` | 否 | mapping | `{}` | 特征选择 + 重采样，仅在训练折上做 |
| `cv` | 否 | mapping | `{n_splits: 5, random_state: 42}` | 交叉验证折数与种子 |
| `fusion` | 否 | mapping | — | 仅多视图 cohort 使用；单视图 pkl 忽略。见 [../usage-multiview-fusion.md](../usage-multiview-fusion.md) |
| `grid` | 否 | mapping | 默认 `TBLS_GRID`/`BLS_GRID` | `--grid` 时要扫的超参轴。见 [grid-search.md](grid-search.md) |
| `output_dir` | 否 | 字符串 | `results_dir` | 运行目录和各 cohort Excel 的输出位置 |

## `model`（选模型）

| 子键 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `name` | 是 | `tbls` | 模型名。`tbls`/`bls` 是包内估计器；其他必须是 [models.md](models.md) 中的基线名。未知名字会抛 `ValueError` 并列出所有支持的名字 |
| 其他键 | 否 | — | 直接传给估计器构造器。`tbls`/`bls` 用 `hyperparams.py` 的默认值；基线接受底层 sklearn 类的任何参数 |
| `random_state` | 否 | `42` | 随机种子，对所有模型生效 |

旧键 `map_num`/`enhance_num` 会自动映射为 `n_map_trees`/`n_enhance_trees`，
兼容老 YAML。

示例：

```yaml
model:
  name: tbls
  n_map_trees: 10
  n_enhance_trees: 10
  use_if_weights: true
  graph_gamma: 0.1
  random_state: 42
```

## `preprocess`（特征选择 + 重采样，仅在训练折上做）

| 子键 | 说明 | 可选值 | 默认值 |
|---|---|---|---|
| `feature_selection` | 在训练折上拟合特征选择器，测试折复用 | `lasso`, `pca`, `mutual_info`, `null` | `null`（不做选择） |
| `resampling` | 在特征选择**之后**对训练折做重采样。测试折从不重采样 | `smote`, `adasyn`, `border_smote`, `undersample`, `tomek`, `smote_tomek`, `smote_enn`, `null` | `null`（不做重采样） |

选择器内部的固定超参（目前不可从 YAML 配——改 `experiments/dataprocess.py`
可调）：

| `feature_selection` | 实现 |
|---|---|
| `lasso` | `Lasso(alpha=0.01)`，保留非零系数对应的特征 |
| `pca` | `PCA(n_components=0.95)` |
| `mutual_info` | `SelectKBest(mutual_info_classif, k=10)` |

重采样器直接对应 imbalanced-learn 中的同名类。

## `cv`（交叉验证）

| 子键 | 说明 | 可选值 | 默认值 |
|---|---|---|---|
| `n_splits` | `KFold` 折数（shuffled） | 整数 ≥ 2 | `5` |
| `random_state` | `KFold(shuffle=True)` 的种子 | 整数 | `42` |

## `output_dir`

相对路径（从仓库根算起）或绝对路径。运行结束后在其下生成：

```
{output_dir}/{run_name}/{timestamp}/                 ← 运行目录（日志 + npz）
{output_dir}/{run_name}/{cohort}/{timestamp}/        ← 各 cohort Excel
```

完整布局见 [outputs.md](outputs.md)。

## `run_name`

可选但建议填写。填了之后，目录名是 `examples/runs/TBLS Full/`，图例显示
`TBLS Full`。不填则退化为 `examples/runs/tbls_biomedical_larger/<timestamp>/`，
图例就是那个自动路径。空格没问题（Windows + openpyxl + 日志路径都能处理）。

## `grid`（仅 `--grid` 时生效）

格式为 `{轴名: [值列表]}`。传 `--grid` 时：

- `model.name: tbls` → 默认扫 `n_map_trees`/`n_enhance_trees`/`reg_param`；
  你在 `grid:` 里写的轴会**替换**同名默认轴，没提到的默认轴**保留**。
- `model.name: bls` → 同理，按 `BLS_GRID` 合并。
- 基线 → 没有默认网格，`--grid` 时 `grid:` **必须**写。

完整语义见 [grid-search.md](grid-search.md)。

## 完整示例

```yaml
# 一次消融运行：完全正则化的 TBLS（GFTBLS）
dataset: biomedical_larger
data_path: examples/datasets/
run_name: TBLS Full

model:
  name: tbls
  n_map_trees: 10
  n_enhance_trees: 10
  use_if_weights: true
  graph_gamma: 0.1
  random_state: 42

preprocess:
  feature_selection: lasso
  resampling: smote

cv:
  n_splits: 5
  random_state: 42

output_dir: examples/runs
```
