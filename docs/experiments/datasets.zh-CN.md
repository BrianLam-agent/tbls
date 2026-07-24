[English](./datasets.md) | 简体中文

# 数据集

`experiments/train.py` 经 `experiments.dataprocess.py::DataLoader` 加载数据
集。把数据集文件放在你的 `data_path` 目录(示例配置用
`examples/datasets/`;`experiments/configs/default.yaml` 用
`experiments/datasets/`)。数据集目录被 git 忽略(大体积二进制),所以你
在本地放 — 永远不会被提交。

## Pickle 格式(你应该只用这个)

对 `.pkl` 跑 `joblib.load(...)` 返回一个 dict。`DataLoader` 接受以下任一:

### 1. 平铺单 cohort
```python
{"data": X, "target": y}
```
- `X` 形 `(n, d)`,`y` 形 `(n,)`。
- 以键 `"single"` 报告。

### 2. 多键(一个 pkl 含若干 cohort)
```python
{"DM": {"data": X_DM, "target": y_DM},
 "CKD": {"data": X_CKD, "target": y_CKD},
 ...}
```
- 每个 value 独立处理,以其 dict 键为 cohort 键。
- `train.py` 遍历每个 cohort;输出按 cohort 键写在对应子目录
  (`{cohort}/{timestamp}/...`)。

### 3. 多视图(CCA/GFCCA 融合)
```python
{"views": {"view_a": X_a, "view_b": X_b, ...}, "target": y}
```
- 自动检测(有 `"views"` 键而不是 `"data"` 键)。
- 需要 YAML `fusion:` 块配 `method`/`view_groups`;见
  [../usage-multiview-fusion.md](../usage-multiview-fusion.md)。单视图 YAML
  完全忽略 fusion。

## DataLoader 做的规范化预处理(已内置,你不用做)

- 标签 `-1` 的样本被丢弃。
- 标签被二值化为 `{0, 1}`(经 `(y > 0).astype(int)`) — 本流水线是二分类
  流水线。
- `dtype=object` 的特征矩阵被强制 `float64`。
- 特征中的 `NaN` 和 `Inf` 被置零。

## 文件放哪

`examples/configs/*.yaml` 设 `data_path: examples/datasets/`。规范副本在
`experiments/datasets/`(测试套件 + smoke run 用);拷到示例位置:

```bash
cp experiments/datasets/biomedical_larger.pkl examples/datasets/
```

本开发机上现有的规范文件见
[`experiments/datasets/README.md`](../../experiments/datasets/README.md)。

## 旧 CSV+label 对(请勿用于二分类 TBLS 实验)

`DataLoader` 还有一条 `_load_csv` 路径,当同时存在 `{dataset}_data.csv`
和 `{dataset}_label.csv` 时触发(它先试 CSV 再试 pkl)。这条路径是为旧的
多标签工作流保留的,用 `MultiLabelBinarizer` — **它不会二值化到 {0, 1}**,
不会丢弃标签 `-1`,也不是 TBLS 训练流水线走的路径。不要在你的 `data_path`
里放 CSV 文件然后期望二分类行为 — 你会静默走到多标签 binarizer 路径。若
你有 CSV 数据,转成上面任一格式的 pkl。
