English | [简体中文](./datasets.zh-CN.md)

# 数据集

`train.py` 通过 `experiments/dataprocess.py::DataLoader` 加载数据集。把
pkl 文件放在 `data_path` 目录下（示例配置用 `examples/datasets/`；
`experiments/configs/default.yaml` 用 `experiments/datasets/`）。数据集
目录已 gitignore（大文件不提交），只在本地放置。

## Pickle 格式（唯一推荐格式）

`joblib.load(...)` 读取 `.pkl`，返回一个 dict。`DataLoader` 接受以下
三种结构：

### 1. 单 cohort
```python
{"data": X, "target": y}
```
- `X` 形状 `(n, d)`，`y` 形状 `(n,)`。
- 以键 `"single"` 报告。

### 2. 多 cohort（一个 pkl 含多个队列）
```python
{"DM": {"data": X_DM, "target": y_DM},
 "CKD": {"data": X_CKD, "target": y_CKD},
 ...}
```
- 每个 value 独立处理，以 dict 键作为 cohort 名。
- `train.py` 遍历每个 cohort，输出按 cohort 名写在对应子目录
  （`{cohort}/{timestamp}/...`）。

### 3. 多视图（CCA/GFCCA 融合）
```python
{"views": {"view_a": X_a, "view_b": X_b, ...}, "target": y}
```
- 自动检测（有 `"views"` 键而非 `"data"` 键）。
- 需要 YAML `fusion:` 块配置 `method`/`view_groups`；见
  [../usage-multiview-fusion.md](../usage-multiview-fusion.md)。单视图
  YAML 完全忽略 fusion。

## DataLoader 自动做的预处理

（已内置，你不需要手动做）

- 标签为 `-1` 的样本被丢弃。
- 标签被二值化为 `{0, 1}`（`(y > 0).astype(int)`）——本流水线是二分类
  流水线。
- `dtype=object` 的特征矩阵被转为 `float64`。
- 特征中的 `NaN` 和 `Inf` 被置零。

## 文件放哪

`examples/configs/*.yaml` 设 `data_path: examples/datasets/`。规范副本
在 `experiments/datasets/`（测试套件和 smoke run 用）；拷到示例位置：

```bash
cp experiments/datasets/biomedical_larger.pkl examples/datasets/
```

本开发机上现有的规范文件见
[`experiments/datasets/README.md`](../../experiments/datasets/README.md)。

## 旧 CSV+label 格式（请勿用于二分类 TBLS 实验）

`DataLoader` 还有一条 `_load_csv` 路径：当同时存在 `{dataset}_data.csv`
和 `{dataset}_label.csv` 时触发（它先试 CSV 再试 pkl）。这条路径是为旧
多标签工作流保留的，用 `MultiLabelBinarizer`——**它不会二值化到 {0, 1}**，
不会丢弃标签 `-1`，也不是 TBLS 训练流水线走的路径。不要在 `data_path`
里放 CSV 文件然后期望二分类行为——你会静默走到多标签路径。有 CSV 数据
的话，先转成上面任一格式的 pkl。
