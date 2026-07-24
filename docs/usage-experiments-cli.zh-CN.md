[English](./usage-experiments-cli.md) | 简体中文

# 运行实验命令行（在真实数据集上训练）

`experiments/` 是用于在你本机以 `tbls` 估计器运行真实数据集的训练/评估/对比流水线。它**不属于**发布的 `tbls` 包（依赖 `pandas`、`imbalanced-learn`、`xgboost`、`openpyxl`、`loguru`、`matplotlib`、`typer`、`pyyaml`——这些重而带主观取向的依赖会使每位 `pip install tbls` 用户被拖累）。缘由见 [`architecture.zh-CN.md`](./architecture.zh-CN.md#3-为什么-srcexperiments-的分裂)。地面起步教程见 [`../examples/README.md`](../examples/README.md)；本文是每个选项、每个产物、每种行为的完整参考。

## 环境搭建

```bash
git clone https://github.com/BrianLam-agent/tbls.git
cd tbls
uv sync --group dev --group experiments
```

确切的 `experiments` 依赖组在 `pyproject.toml` 的 `[dependency-groups] experiments = [...]` 中声明。用 `uv`: `--group experiments` 装它；本地建议也装 dev 组（`pytest`/`ruff`/`mypy`）。

## 数据集

把数据集文件置于 `experiments/datasets/` 下（该目录被 git 忽略——见 [`experiments/datasets/README.md`](../experiments/datasets/README.md)）。训练 CLI 通过 `experiments/dataprocess.py::DataLoader` 加载。

### 支持的加载器（按文件存在性自动选择）

`DataLoader` 先试 `.csv`，再退回 `.pkl`：

1. **CSV + 标签 CSV 配对** — 当 `{dataset}_data.csv` 与 `{dataset}_label.csv` 都存在时，用 `np.loadtxt` 加载（`float32` 的 X、`int32` 的标签）。此路径服务于旧的多标签工作流，**不**丢弃标签 `-1`、不二值化，而用 `MultiLabelBinarizer`。**不要**用此路径跑 `TBLS` 的二分类实验；它是旧数据入口钩子。
2. **Pickle（`tbls` 二分类流水线默认）** — 当 `{dataset}.pkl` 存在时，`joblib.load` 将其当作以下之一：

   - 扁平 `{"data": X, "target": y}` 字典（按键 `"single"` 报告），
   - 此类子数据集字典的多键字典（按各值独立处理——例如一个文件容纳多个疾病队列 `{"DM": {...}, "CKD": {...}, ...}`），
   - 多视图字典（仍按键区分）以 `{"views": {...}, "target": y}` 代替 `"data"` ——自动检测为多视图融合（见 [`usage-multiview-fusion.zh-CN.md`](./usage-multiview-fusion.zh-CN.md)）。

   pkl 路径以规范方式预处理标签：丢弃 `y == -1` 的样本、二值化为 `{0, 1}`（`(y > 0).astype(int)`）、`dtype=object` 的特征矩阵被强制转为 `float64`、`NaN`/`Inf` 置零。与 `experiments/smoke_run.py::_extract_xy` 同一套规范化。

## 最小健全性检查：`smoke_run.py`

`experiments/smoke_run.py` 是确认数据集正确加载、`TBLS` 能拟合+预测正常的最快途径（小模型、一次训练/测试划分、一组断言：`predict_proba` 有限、行和近似 1、非单类预测）：

```bash
uv run --group experiments python experiments/smoke_run.py
```

```
TBLS smoke check OK | key=DM acc=0.9208 macro_f1=0.8985 bls_acc=0.8387 train=1362 test=341 features=204
```

`run_smoke_check(pkl_path, key=None, max_rows=2000, random_state=42)` 入参见模块文档；可通过直接导入指向任意 pkl。

## 完整训练命令行：`train.py`

```bash
# 单配置
uv run --group experiments python experiments/train.py --config experiments/configs/default.yaml
# 用 CLI 覆盖配置
uv run --group experiments python experiments/train.py --dataset biomedical_larger --n-splits 3
# 批量：依次跑目录里每个 *.yaml/*.yml
uv run --group experiments python experiments/train.py --config-dir examples/configs --n-splits 5
```

### CLI 选项

| 选项 | 覆盖 | 含义 |
|---|---|---|
| `--config PATH` | — | YAML 配置路径（默认 `experiments/configs/default.yaml`）。 |
| `--config-dir DIR` | — | 依次跑 `DIR` 内每个 `*.yaml`/`*.yml`（按名排序）。每个配置有独立 run 目录、JSONL 日志与 `.npz` 侧文件。CLI 覆盖对批次里每个配置都生效。与 `--config` 互斥。 |
| `--dataset NAME` | `dataset` | 数据集词干；加载 `{data_path}/{NAME}.pkl`。 |
| `--model NAME` | `model.name` | 模型。内置层 `tbls`/`bls`；**其他名称**均派发到 `experiments.classifiers.create_classifier`（见下文"基线模型"）。 |
| `--map-num N` | `model.map_num` | 旧别名 `TBLS(n_map_trees=N)`（仅 TBLS）。 |
| `--n-splits N` | `cv.n_splits` | `KFold` 折数。 |
| `--output-dir DIR` | `output_dir` | run + cohort 产物写在哪（见"产物布局"）。 |
| `--fusion [cca\|gfcca]` | `fusion.method` | 覆盖多视图队列的融合方法（单视图队列忽略）。 |
| `--grid` | — | 扫描模型超参网格；见下文"网格搜索"。默认网格 `TBLS_GRID`/`BLS_GRID` 在 `hyperparams.py`，可用 YAML `grid:` 覆盖。仅对 `tbls`/`bls` 有效；基线需要 YAML `grid:`，否则降级为单 k-fold 并告警。 |

## YAML 配置参考

```yaml
dataset: biomedical_larger          # pkl 词干(加载 {data_path}/{dataset}.pkl)
data_path: examples/datasets/       # 存放 pkl 的目录；默认 experiments/datasets/

run_name: TBLS Full                 # 人选实验名。成为 run 目录词干以及 JSONL
                                    # 与图例的 label。可选；缺省回退到
                                    # {model.name}_{dataset}/{timestamp}。

model:                              # 每个 model.name 一套规则:
  name: tbls                        #   'tbls'/'bls' -> 包内估计器，配合
                                    #     hyperparams.TBLS_DEFAULTS/BLS_DEFAULTS。
                                    #     YAML 锢(经旧键映射 map_num->n_map_trees、
                                    #     enhance_num->n_enhance_trees)只覆盖合法
                                    #     构造器参。
                                    #   其他名 -> create_classifier(name, ...);
                                    #     YAML 锢变 **kwargs；random_state 从
                                    #     model.random_state 读(默认 42)。
  n_map_trees: 10                   # TBLS: 映射树数
  n_enhance_trees: 10               # TBLS: 增强树数
  use_if_weights: true              # TBLS: 打开 IFS 样本权重(其与 BLS 的区别点)
  graph_gamma: 0.1                  # TBLS: 图-拉普拉斯正则强度(0=关)
  random_state: 42

preprocess:
  feature_selection: lasso          # lasso | pca | mutual_info | null
  resampling: smote                 # smote | adasyn | border_smote | undersample |
                                    # tomek | smote_tomek | smote_enn | null
                                    # (仅作用训练折，特征选择之后)

cv:
  n_splits: 5                       # KFold 折数
  random_state: 42

fusion:                             # 仅多视图队列(pkl 有 "views" 非 "data")相关；
  method: gfcca                     #   单视图队列忽略整段。
  view_groups:                      # 详见 usage-multiview-fusion.zh-CN.md。
    - ["view_a", "view_b"]

output_dir: examples/runs          # run + cohort 输出写在这

grid:                               # 可选 {-轴名 -> 取值列表}。
  use_if_weights: [false, true]    # 存在时，这里命名的轴 精确替换 默认同名轴
  graph_gamma: [0.0, 0.05, 0.1]    # (YAML 列表胜)；默认里 未命名的轴 被保留 (覆盖/扩展语义)。
                                    # 基线无默认网格，故要扫基线必须给 YAML `grid:`。
```

### `feature_selection` 内部细节

| 取值 | 实现(在 `experiments/dataprocess.py`) |
|---|---|
| `lasso` | `Lasso(alpha=0.01)`；保留非零系数的特征，该 mask 在测试折上复用。 |
| `pca` | `PCA(n_components=0.95)`；`transform` 同时作用于训练与测试折。 |
| `mutual_info` | `SelectKBest(mutual_info_classif, k=10)`；`transform` 同样就地施加。 |
| `null` / 缺省 | 不做特征选择。 |

内部固定的 `alpha=0.01` / `n_components=0.95` / `k=10` 当前不可从 YAML 调；若需改，编辑 `experiments/dataprocess.py::FEATURE_SELECTORS` 常量或直接调 `DataLoader`。

### `resampling` 内部细节

| 取值 | imbalanced-learn 类 |
|---|---|
| `smote`, `adasyn`, `border_smote` | `SMOTE`、`ADASYN`、`BorderlineSMOTE` （过采样）。 |
| `undersample`, `tomek` | `RandomUnderSampler`、`TomekLinks` （下采样）。 |
| `smote_tomek`, `smote_enn` | `SMOTETomek`、`SMOTEENN` （组合）。 |
| `null` / 缺省 | 不做重采样。 |

所有重采样器仅作用训练折（特征选择之后）；测试折不动。多视图队列的 `MultiViewDataLoader` 拒绝需要单视图的 `SMOTETomek`/`SMOTEENN` 等 SMOTE 系——见 `usage-multiview-fusion.zh-CN.md`（其约束来自按视行对齐）。

## 产物布局

每次 `train.py` 运行会在 `output_dir`（默认回退 `results_dir/`）下产出：

```
{output_dir}/{run_name}/{timestamp}/                 <- run 目录(loguru + npz)
    logs/{dataset}_{timestamp}.jsonl                  <- 结构化日志(见下文)
    logs/{dataset}_{timestamp}_{cohort}_predictions.npz
                                                     <- 原始逐折预测
                                                        (仅非 grid 运行)

{output_dir}/{run_name}/{cohort}/{timestamp}/        <- cohort xlsx 目录
    {cohort}_{model_name}_FS-{...}_RS-{...}.xlsx     <- 每 cohort 一份 xlsx
```

其中 `{timestamp}` 是 `time.strftime("%Y%m%d_%H%M%S")`，由两分支共享（带 `.npz` 的 run 目录与 cohort xlsx 目录是按相同时间戳平铺在 `run_name` 下）。应用级输出目录 `examples/runs/`/`plots/` 都被 git 忽略——与 `dist/`、`.pytest_cache/` 同处理。

### Excel 工作表布局

每 cohort 的 xlsx (`{cohort}_{model_name}_FS-..._RS-..._.xlsx`) 含，见 `TBLSResultSaver`:

- **非 grid 运行**：`{model}_Details` 工作表(每折一行) + `{model}_Summary` 工作表(一行，跨折均值在 `avg_*` 键前缀下，附 cohort 键)。
- **`--grid` 运行**：
  - 每 grid 点一份 `Grid_{i:03d}` 工作表（该点 `_cross_validate` 的逐折行），
  - `GridSummary` 工作表（每 grid 点一行，按 `avg_balanced_accuracy` 降序排列；含被扫超参、所有 `avg_*` 指标、外加 `rank` 与 `is_winner`——**第 1 行 winner**，`is_winner=True`），
  - `GridSearchLog` 工作表（与 `GridSummary` 内容相同，flat pre-sort 顺序，读作时序搜索日志）。

文件中元数据工作表 `{sheet}_Meta`（`save_summary` 写）记录所用 `Feature_Selection` 与 `Resampling_Method`，便于追溯。

## 结构化 JSONL 日志

`train.py` 在运行开始时调用 `experiments.logging_setup.configure_logging`，移除 loguru 默认 sink 并加入两个：

- 人类可读的 **stdout** sink，level `INFO`（彩色 `<level>{level: <8}</level> {message}` 格式），保持以往的目阅输出；
- 结构化 **JSONL 文件** sink，level `DEBUG`，`serialize=True`，每行一个 JSON 对象，写到 `logs/{dataset}_{timestamp}.jsonl`。stdlib `logging` 经 `InterceptHandler` 拦截，使仍在用 `logging.getLogger` 的模块（如 `experiments.evaluate` 的概率指标告警）也流进 JSONL。

### 事件 schema

每行事件含 loguru 的 record 元数据（`record.text`、`record.level`、`record.time`、`record.function`、`record.line` 等）写在 `"record"`，bound event 的 payload 在 `record["extra"]`。有类型的 schema 住在 `experiments/logging_schema.py`（`RunStartedEvent`、`FoldCompletedEvent`、`GridPointCompletedEvent`、`GridSummaryEvent`、`RunFinishedEvent`）——这些 `TypedDict` 是规范描述，下表只列判别符与关键字段：

| 事件 | 触发 | `record.extra` 关键字段 |
|---|---|---|
| `run_started` | 每 run 一次 | `dataset`、`model`、`fusion_method`、`grid`、`run_name`（可选） |
| `fold_completed` | 每折+每 cohort | `dataset`、`cohort_key`、`fold`、`n_splits`、`metrics`（`MetricsDict`）、`grid_idx`/`grid_params`（仅 `--grid`）、`predictions_file` |
| `grid_point_completed` | `--grid` 每 grid 点 | `dataset`、`cohort_key`、`grid_idx`、`n_grid_points`、`grid_params`、`metrics`（`avg_*` 前缀均值） |
| `grid_summary` | `--grid` 排后每 cohort 一次 | `dataset`、`cohort_key`、`winner_params`、`winner_metric`、`n_grid_points` |
| `run_finished` | 每 run 一次 | `dataset`、`duration_seconds` |

标量 `metrics` 字典遵循 `experiments/metrics_schema.py` 中的 `MetricsDict` TypedDict（二值路径：accuracy/precision/recall/f1/... 加上 additively 的 MCC/Kappa/log_loss/brier_score；多类路径：macro/weighted 均值与一对多 specificity/NPV/gmean）。当 `y_score` 缺失或失败时，`auroc`/`auprc`/`optimal_threshold`/`log_loss`/`brier_score` 降级为 `None`；多类下除 `auroc` 外这五个均缺省。

### `.npz` 预测侧文件

为使 JSONL 只装标量，原始逐折 `y_true`/`y_pred`/`y_score` 数组被存到 `logs/{dataset}_{timestamp}_{cohort}_predictions.npz`（非 grid 运行每 cohort 一份），键为 `{cohort}_fold{N}_{y_true,y_pred,y_score}`。`y_score` 是 `model.predict_proba(x_te)`，`float32`、shape `(n_test, n_classes)`——**不是** 1-D 正类概率；要正类必须切 `y_score[:, 1]`（visualize/compare CLI 已这么做）。仅非 grid 运行产出（27×n_folds×n_cohorts 的 grid 会撑爆侧文件）；`fold_completed` 事件的 `predictions_file` 字段命名它所在的侧文件（grid 运行/未持久化折为 `None`）。

## 超参默认与网格搜索

默认值与网格轴住在 `experiments/hyperparams.py`，皆普通可编辑的 Python 字典：

| 字典 | 用于 | 当前值 |
|---|---|---|
| `TBLS_DEFAULTS` | `model.name: tbls` 单 run 默认值（与 YAML `model:` 覆盖合并） | `n_map_trees: 10, n_enhance_trees: 10, tree_max_depth: 5, tree_min_samples_split: 3, tree_max_features_ratio: 0.7, reg_param: 1e-8`（`graph_strategy`/`if_strategy` 留在 TBLS 构造器默认值 `discriminative`/`simple`） |
| `BLS_DEFAULTS` | `model.name: bls` 单 run 默认值 | `n_feature_groups: 30, n_feature_nodes_per_group: 40, n_enhancement_groups: 1, n_enhancement_nodes_per_group: 500, reg_param: 1.0, map_func/enhance_func: "relu"` |
| `TBLS_GRID` | `--grid` 对 `tbls` 扫的轴（或 YAML `grid:` 与其合并，见上） | `n_map_trees: [10, 20, 40], n_enhance_trees: [10, 20, 40], reg_param: [1e-8, 1e-4, 1e-2]`（3×3×3 = 27 点默认） |
| `BLS_GRID` | `--grid` 对 `bls` 扫的轴 | `n_feature_groups: [15, 30, 60], n_feature_nodes_per_group: [20, 40, 80], reg_param: [0.1, 1.0, 10.0]`（27 点） |
| `CCA_DEFAULTS`/`CCA_GRID`/`GFCCA_DEFAULTS`/`GFCCA_GRID` | 多视图队列的融合超参（`CCA_GRID`/`GFCCA_GRID` 在本版本**不**被 `--grid` 扫） | 详见 `usage-multiview-fusion.zh-CN.md` |

### `--grid` 语义（两层决策，见 `train.py::_resolve_grid`）

1. **默认网格** = `TBLS_GRID`（`model.name: tbls`）或 `BLS_GRID`（`model.name: bls`）。要改默认扫什么，编辑 `hyperparams.py` 里的模块级字典（它特意是 Python 常量、不进 YAML——所以值在代码评审里被记录）。
2. **YAML `grid:` 覆盖/扩展**——存在时，YAML 命名的轴**精确替换**默认同名轴（YAML 列表胜）；默认中**未命名**的轴被**保留**。这样 `grid:` 能收缩默认、换一个轴、或新增一轴（例如 `use_if_weights: [false, true]` 去扫一个先前固定的标志）。
3. **基线（`lr`/`rf`/...）无默认网格**——必须给 YAML `grid:`。无 YAML 的基线跑 `--grid` 会抛清晰 `ValueError("No default grid ... Set YAML grid: ...")`。
4. 给基线传 `--grid` 但**没有** YAML `grid:` 退化为一次 k-fold，并打告警（**不**静默去扫假网格）。

### 网格产物

每个 `--grid` cohort：

- `Grid_{i:03d}` 工作表（按点、逐折行），
- `GridSummary`（排序、第 1 行 `is_winner` 标记），
- `GridSearchLog`（flat），
- `grid_point_completed` JSONL 事件（每点一个），
- 收尾一个 `grid_summary` 事件，含 `winner_params`/`winner_metric`/`n_grid_points`。

`experiments/visualize.py` 读回这些产出 `grid_search_summary.png`（指标 vs 每个被扫轴，每轴一个子图）。

## 基线模型（任意 `experiments.classifiers.create_classifier` 中的模型）

`model.name: <baseline>` 派发到 `experiments.classifiers.create_classifier(name, random_state, **kwargs)`。这是为将 `TBLS`/`BLS` 与标准基线对垒而早已存在的工厂。支持名含：`rf`、`svm`、`xgb`、`lgb`、`catboost`、`knn`、`lr`、`lasso`、`elasticnet`、`nb`、`lda`、`cart`、`mlp`、`dnn`、`extratrees`、`gbdt`、`block_plsda`、`block_splsda`、`mogonet`、`mogonet_nn`、`mofa`、`diablo`、`snf`（完整规范列表在 `experiments/classifiers.py` 的 `create_classifier` docstring）。

约定：

- **类失衡处理**：内置。每个被支持基线均以 `class_weight="balanced"`（或等价）构造，故即便 `--resampling: null` 也得到类均衡拟合。
- **软依赖**：`xgboost` 在 `experiments` 依赖组内；`lightgbm`、`catboost`、`torch` **不在**（按需懒加载，仅当请求对应 `model.name` 且未安装时由 `create_classifier` 抛清晰 `ImportError`）。
- **YAML `model:` 键变 `**kwargs`** 进入底层估计器；只 `name` 与 `random_state` 被特殊处理。无 YAML `grid:` 时 `--grid` 对基线无效。

## 对比与可视化 CLI

### `visualize.py —— per-fold / grid-search / ROC / PR / confusion 图

各 `--dir` 经 `experiments/run_resolution.py::resolve_run_dir` 解析（见"`--dir` 解析规则"），读 JSONL 的 `fold_completed` + `grid_point_completed`（+ `grid_summary`）事件。产出（`--output-dir` 下，默认首个 `--dir` 旁的 `plots/`）：

| 文件 | 来源 | 范围 |
|---|---|---|
| `per_fold_metrics.png` | 标量 `fold_completed` 指标 | 始终 |
| `grid_search_summary.png` | `grid_point_completed` 行 | 仅 `--grid` 运行 |
| `roc_{cohort}.png` | `.npz` 侧文件的 `y_true`/`y_score`（正类按 `[:, 1]`） | 仅非 grid 运行，**每 cohort 一张 PNG** |
| `pr_{cohort}.png` | 同上 | 同样——**每 cohort 一张 PNG** |
| `confusion_{run}.png` | `.npz` 侧文件的 `y_true`/`y_pred` | 仅非 grid 运行，每 run 一张 PNG（cohort 作子图） |

ROC 与 PR 按 cohort 分开是设计选择——">消融对比=同一 cohort 下 run A vs run B"，每 cohort 文件叠加所有 run；per-fold 柱状与 grid-search 摘要保持原布局。

#### `--dir` 解析规则

`visualize.py` 与 `compare.py` 一致。`--dir` 参数可为：

- **run-name 层**（如 `examples/runs/TBLS Full`）——CLI 自动选其下最新的 `YYYYMMDD_HHMMSS` 时间戳子目录；或
- **run-name/<timestamp>** 层（如 `examples/runs/TBLS Full/20260724_074140`）——直接用。

再深（如 `.../<timestamp>/logs`）、再浅（如 `examples/runs`）、或子目录名不是 `YYYYMMDD_HHMMSS` 的均**报错**带清晰诊断——不靠 shell glob、不静默选旧 run。run name 可含空格（Sheet 名、路径、legend label 均保空格）。`compare.py` 额外用 `run_resolution.cohort_excel_dir` 在**同**时间戳下找 sibling 的 cohort xlsx 目录——若时间戳不一致也以同款错误拒绝。

#### `visualize.py` CLI 选项

| 选项 | 默认 | 含义 |
|---|---|---|
| `--dir DIR [`--dir DIR ...`]` | — | 一或多个 run 目录（按上文规则解析）。 |
| `--output-dir DIR` | 首个 `--dir` 旁的 `plots/` | PNG 写在哪。 |
| `--dpi N` | `300` | PNG 分辨率。`--dpi 120` 用于快速预览。 |

### `compare.py —— 跨 run 对比 Excel

每个 `--dir` 同同款规则解析，跨 run 解析每个 `fold_completed` 事件，在 `--output-dir`（默认 `examples/comparison`）下写 `comparison.xlsx`：

- **每 cohort 一 sheet** + 一张 `README` sheet 说明布局。
- 行 = run（每个 `--dir` 一个，排序）；列 = `ORDERED_METRICS` 次序下的 15 个标量指标（`balanced_accuracy`、`accuracy`、`f1_score`、`mcc`、`cohen_kappa`、`auroc`、`auprc`、`recall`、`specificity`、`precision`、`negative_predictive_value`、`gmean`、`hamming_loss`、`log_loss`、`brier_score`）。
- 每格 `mean (std)`（跨 CV 折）。`--no-std` 退化为纯均值。
- **加粗** = 该指标在 cohort 上最好的那个 run，方向由 `METRIC_DIRECTION` 决定（越大越好的：`auroc`/`balanced_accuracy`/`mcc`/`cohen_kappa`/`accuracy`/`f1_score`/`recall`/`specificity`/`precision`/`negative_predictive_value`/`gmean`/`auprc`；越小越好的：`hamming_loss`/`log_loss`/`brier_score`）。
- 一个 run 若未生成某 cohort，对应格留空（不是 0/NaN）。

#### `compare.py` CLI 选项

| 选项 | 默认 | 含义 |
|---|---|---|
| `--dir DIR [`--dir DIR ...`]` | — | 一或多个 run 目录（经 `resolve_run_dir` 解析）。 |
| `--output-dir DIR` | `examples/comparison` | `comparison.xlsx` 写在哪。 |
| `--no-std` | 关 | 不附 `(std)`；写纯均值代替 `mean (std)`。 |

## 读图（未标定 TBLS 输出在 PR 上的腰斩）

TBLS（及 `TBLS Full`/`TBLS Graph`/`TBLS IFS`）的 `predict_proba` 为 ridge 回归输出 `Z = W A_{enh}` 经 softmax 转换——**不带任何概率标定步骤**（无 Platt 标定、无 sigmoid/isotonic 标定）。某些数据集上，大批低置信样本评分被挤到 `0.5` 附近的窄峰里，PR 阈值扫描穿越该密坪时整块样本齐齐"预测为正"，而其中只有 ~`prevalence` 才是真阳，于是 precision 暴跌至 `prevalence`、recall 跃升——PR 图上即一道近乎垂直的悬崖。详见 [`usage-figures-and-calibration.zh-CN.md`](./usage-figures-and-calibration.zh-CN.md)：完整数学、复现脚本、建议的缓解（Platt 标定或 grouped-bin calibrator 当作 future work，非当前实现 bug）。

LR 的概率直接来自凸 log-loss 最优解，所以它沿 `[0, 1]` 平滑分布、没有 `0.5` 密坪——正是同图里 LR 曲线平滑、TBLS 曲线"诡异"的原因。**ROC 曲线所受影响小得多**，因为阈值分布变化对 TPR/FPR 推动更平滑（ROC 积分对单调的 score-reshaping 不变）。

## 多视图队列的 `--grid`

`--grid` 只扫模型网格（`TBLS_GRID`/`BLS_GRID` 或 YAML `grid:`）。融合超参（`hyperparams.py` 的 `CCA_GRID`/`GFCCA_GRID`）在本版本中**不**被 `--grid` 扫——明确的范围限制，不静默跳过。缘由（多视图 pkl 契约、按视行对齐/重采样约束）见 [`usage-multiview-fusion.zh-CN.md`](./usage-multiview-fusion.zh-CN.md)。

## 命令速查表

| 需要 | 运行 |
|---|---|
| 一行数据集+模型健全性 | `uv run --group experiments python experiments/smoke_run.py` |
| 从配置跑单次 TBLS | `uv run --group experiments python experiments/train.py --config examples/configs/tbls_full.yaml` |
| 批跑目录下每个配置 | `uv run --group experiments python experiments/train.py --config-dir examples/configs --n-splits 5` |
| 消融图叠加 | `uv run --group experiments python experiments/visualize.py --dir "examples/runs/TBLS" --dir "examples/runs/TBLS Full" ...` |
| 对比 Excel（mean (std)、最优加粗） | `uv run --group experiments python experiments/compare.py --dir "examples/runs/TBLS" ...` |
| 网格搜索 | `uv run --group experiments python experiments/train.py --config examples/configs/tbls_grid.yaml --grid` |