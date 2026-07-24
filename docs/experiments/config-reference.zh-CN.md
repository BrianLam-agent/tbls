[English](./config-reference.md) | 简体中文

# YAML 配置参考

YAML 配置驱动一次 `experiments/train.py` 运行。本页列出每个键、干啥、
期望什么值、不写时的默认、以及一个最小示例。*哪些 `model.name` 合法*见
[models.md](models.md);哪些 CLI 选项可覆盖这些见
[cli-train.md](cli-train.md)。

## 顶层键

| 键 | 必填? | 类型 | 默认 | 用途 |
|---|---|---|---|---|
| `dataset` | 是 | 字符串 | — | 数据集名;加载 `{data_path}/{dataset}.pkl`。 |
| `data_path` | 否 | 字符串 | `experiments/datasets` | 存放 pkl 的目录。 |
| `run_name` | 否 | 字符串 | `{model.name}_{dataset}` | 同时作运行目录段名和图表/Excel 标签。写好以得清晰标签;允许空格。 |
| `model` | 是 | mapping | `{name: tbls}` | 选模型 + 其构造器 kwargs。 |
| `preprocess` | 否 | mapping | `{}` | 特征选择 + 重采样,只在训练折做。 |
| `cv` | 否 | mapping | `{n_splits: 5, random_state: 42}` | 交叉验证折数与种子。 |
| `fusion` | 否 | mapping | — | 仅多视图 cohort 用;单视图 pkl 忽略。见 [../usage-multiview-fusion.md](../usage-multiview-fusion.md)。 |
| `grid` | 否 | mapping | (默认 `TBLS_GRID`/`BLS_GRID`) | `--grid` 下要扫的超参轴。见 [grid-search.md](grid-search.md)。 |
| `output_dir` | 否 | 字符串 | `results_dir` | 运行目录 + 各 cohort Excel 写到哪。 |

## `model`(选估计器)

| 子键 | 必填? | 默认 | 效果 |
|---|---|---|---|
| `name` | 是 | `tbls` | 选哪个估计器。`tbls`/`bls` 是包内的;其他必须是 [models.md](models.md) 里的基线名。未知名字会抛 `ValueError` 并列出支持集。 |
| 其他任何键 | 否 | — | 作为构造器 kwarg 转发给估计器(`tbls`/`bls` 用 `hyperparams.py` 的默认;基线接受底层 sklearn 类接受的任何 kwarg)。 |
| `random_state` | 否 | `42` | 种子;对 `tbls`/`bls` 和所有基线生效(两层都从此读 base `random_state`)。 |

旧键 `map_num`/`enhance_num` 别名到 `n_map_trees`/`n_enhance_trees`,
兼容老 YAML。

示例:

```yaml
model:
  name: tbls
  n_map_trees: 10
  n_enhance_trees: 10
  use_if_weights: true
  graph_gamma: 0.1
  random_state: 42
```

## `preprocess`(特征选择 + 重采样,仅训练折)

| 子键 | 效果 | 可选值 | 默认 |
|---|---|---|---|
| `feature_selection` | 选一个特征选择器,在训练折拟合,测试折复用。 | `lasso`,`pca`,`mutual_info`,`null` | `null`(不选) |
| `resampling` | 选一个 imbalanced-learn 采样器,在特征选择**之后**应用于训练折。测试折从不重采样。 | `smote`,`adasyn`,`border_smote`,`undersample`,`tomek`,`smote_tomek`,`smote_enn`,`null` | `null`(不重采样) |

选择器的内部固定超参(目前不可由 YAML 配 — 改 `experiments/dataprocess.py`):

| `feature_selection` | 实现 |
|---|---|
| `lasso` | `Lasso(alpha=0.01)`;留非零系数。 |
| `pca` | `PCA(n_components=0.95)`。 |
| `mutual_info` | `SelectKBest(mutual_info_classif, k=10)`。 |

重采样器直接对应同名 imbalanced-learn 类。

## `cv`(交叉验证)

| 子键 | 效果 | 可选值 | 默认 |
|---|---|---|---|
| `n_splits` | `KFold` 折数(shuffled)。 | 整数 ≥ 2 | `5` |
| `random_state` | `KFold(shuffle=True)` 的种子。 | 整数 | `42` |

## `output_dir`

相对(从仓库根)或绝对目录。CLI 会在其下创建:

```
{output_dir}/{run_name}/{timestamp}/                 <- 运行目录 (日志 + npz)
{output_dir}/{run_name}/{cohort}/{timestamp}/        <- 各 cohort Excel
```

完整布局见 [outputs.md](outputs.md)。

## `run_name`

可选但建议。设了的话,你的目录是 `examples/runs/TBLS Full/`,图例显示
`TBLS Full`。不设的话,退化为 `examples/runs/tbls_biomedical_larger/
<timestamp>/`,图例就是那个自动路径。空格没问题(Windows + openpyxl + log
路径都接受)。

## `grid`(只有 `--grid` 时有用)

一个 `{轴名: [值列表]}` 的 mapping。当你传 `--grid`:

- `model.name: tbls` → 默认扫的轴是 `n_map_trees` / `n_enhance_trees` /
  `reg_param`;你在 `grid:` 里给的任何东西会*替换*同名默认轴,你没提到的
  默认轴*保留*。
- `model.name: bls` → 同样按 `BLS_GRID` 合并。
- 基线 → 没默认网格;`--grid` 时 `grid:` *必须*。

完整语义:[grid-search.md](grid-search.md)。

## 完整注释示例

```yaml
# 一次消融运行:完全正则化的 TBLS (GFTBLS)
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
