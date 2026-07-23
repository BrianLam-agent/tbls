[English](./usage-experiments-cli.md) | 简体中文

# 运行实验命令行（在真实数据集上训练）

`experiments/` 是用于在你本机以 `tbls` 估计器运行真实数据集的训练/评估流水线。它**不属于**发布的 `tbls` 包（依赖 `pandas`、`imbalanced-learn`、`xgboost`、`typer`、`pyyaml`、`openpyxl`--这些重而带主观取向的依赖否则会拖累每位 `pip install tbls` 的用户）。缘由见 [`architecture.md`](./architecture.zh-CN.md)。

## 环境搭建

```bash
git clone https://github.com/BrianLam-agent/tbls.git
cd tbls
uv sync --group dev --group experiments
```

将数据集 `.pkl` 文件置于 `experiments/datasets/` 下（该目录被 git 忽略--见 [`experiments/datasets/README.md`](../experiments/datasets/README.md)）。期望的 pkl 形态为以下之一：

- 扁平的 `{"data": X, "target": y}` 字典，或
- 此类子数据集字典的多键字典（每个值按其键独立处理--例如一个文件容纳多个疾病队列）。

标签为 `-1` 的样本被丢弃；标签被二值化为 `{0, 1}`（`(y > 0).astype(int)`），与旧版流水线约定一致。以 `dtype=object` 存储的特征矩阵被强制转为 `float64`，`NaN`/`Inf` 值置零。

## 最小健全性检查：`smoke_run.py`

在运行完整命令行之前，`experiments/smoke_run.py` 是确认数据集正确加载、`TBLS` 能在其上正常拟合与预测的最快途径（小模型、一次训练/测试划分、若干断言--`predict_proba` 有限、各行概率和为 1、预测非退化）：

```bash
uv run --group experiments python experiments/smoke_run.py
```

```
TBLS smoke check OK | key=DM acc=0.9208 macro_f1=0.8985 bls_acc=0.8387 train=1362 test=341 features=204
```

默认加载 `experiments/datasets/biomedical_larger.pkl`。修改 `smoke_run.py::main()` 中的 `pkl_path`（或直接导入 `run_smoke_check`）以指向其他文件：

```python
from pathlib import Path
from experiments.smoke_run import run_smoke_check

result = run_smoke_check(Path("experiments/datasets/data_cross_train.pkl"), max_rows=2000)
print(result)
```

## 完整训练命令行：`train.py`

```bash
uv run --group experiments python experiments/train.py
uv run --group experiments python experiments/train.py --dataset biomedical_larger --n-splits 3
uv run --group experiments python experiments/train.py --config experiments/configs/default.yaml --map-num 20
```

| 选项 | 覆盖的配置键 | 含义 |
|---|---|---|
| `--config PATH` | - | YAML 配置路径（默认 `experiments/configs/default.yaml`）。 |
| `--dataset NAME` | `dataset` | 数据集名 stem；加载 `{data_path}/{NAME}.pkl`。 |
| `--map-num N` | `model.map_num` | `TBLS(n_map_trees=N)`。 |
| `--n-splits N` | `cv.n_splits` | `KFold` 折数。 |
| `--output-dir DIR` | `output_dir` | Excel 结果写入位置。 |

`experiments/configs/default.yaml`：

```yaml
dataset: biomedical_larger
data_path: experiments/datasets/

model:
  name: tbls
  map_num: 10
  enhance_num: 10
  reg_param: 2.0e-15

preprocess:
  feature_selection: lasso   # lasso | pca | mutual_info | null
  resampling: smote          # smote | adasyn | border_smote | undersample |
                              # tomek | smote_tomek | smote_enn | null

cv:
  n_splits: 5
  random_state: 42

output_dir: results_dir
```

### `train.py` 对每个子数据集键的处理

1. 加载 pkl（所有子数据集键，或键 `"single"` 下的单一扁平字典）。
2. 对每个键，运行 `sklearn.model_selection.KFold`（打乱，`cv.random_state`）。
3. 每折：**仅在训练折上**拟合 `experiments.dataprocess.DataLoader` 的特征选择 + 重采样（不泄漏至测试折），拟合 `tbls.TBLS`，以 `experiments.evaluate.TBLSEvaluator.calculate_metrics` 评估（accuracy、precision、recall、F1、specificity、balanced accuracy、g-mean、AUROC、AUPRC、最优阈值）。
4. 经 `experiments.evaluate.TBLSResultSaver` 将逐折结果与跨折平均写入 `{output_dir}/tbls_{dataset}/{key}/{timestamp}/{key}_tbls_FS-{...}_RS-{...}.xlsx`。

示例日志输出（一个 pkl 中四个子数据集，2 折 CV）：

```
INFO dataset=biomedical_larger keys=['DM', 'CKD', 'BC', 'CG']
INFO === biomedical_larger / DM : X=(1703, 204) y=(1703,) ===
INFO dataset=biomedical_larger key=DM fold=1/2 acc=0.9085
INFO dataset=biomedical_larger key=DM fold=2/2 acc=0.9166
INFO dataset=biomedical_larger key=DM avg={'avg_accuracy': 0.9125, ...}
```

## 特征选择与重采样选项

来自 `experiments/dataprocess.py::DataLoader`：

| `feature_selection` | 实现 |
|---|---|
| `lasso` | `sklearn.linear_model.Lasso(alpha=0.01)`；选取非零系数，掩码在测试折上复用。 |
| `pca` | `sklearn.decomposition.PCA(n_components=0.95)`。 |
| `mutual_info` | `SelectKBest(mutual_info_classif, k=10)`。 |
| `null` / 省略 | 不做特征选择。 |

| `resampling` | 实现 |
|---|---|
| `smote`、`adasyn`、`border_smote` | 过采样（`imblearn.over_sampling`）。 |
| `undersample`、`tomek` | 欠采样（`imblearn.under_sampling`）。 |
| `smote_tomek`、`smote_enn` | 组合（`imblearn.combine`）。 |
| `null` / 省略 | 不做重采样。 |

重采样在特征选择之后、且仅作用于训练折。

## 对比分类器

`experiments/classifiers.py` 是一个大型工厂（`rf`、`svm`、`xgb`、`knn`、`lr`、`cart`、`mlp`、`extratrees`、`gbdt`、`bls`、`tbls` 等--完整列表见模块文档字符串），用于将 `TBLS`/`BroadLearningSystem` 与标准基线对照。它引用的可选依赖（`lightgbm`、`catboost`、`torch`、`muon`）为软依赖--每个都由各自的 `try/except ImportError` 保护，故若未安装，工厂会优雅降级；`uv sync --group experiments` 仅安装 `tbls` 自身训练流水线所需的依赖（含 `xgboost`，不含 `lightgbm`/`catboost`/`torch`）。

## 结果去向

`results_dir/`（或你配置的 `output_dir`）被 git 忽略--见根 `.gitignore`。其下任何内容均不应提交；将其视为临时输出，与 `dist/`、`.pytest_cache/` 等同等对待。
