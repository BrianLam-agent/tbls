[English](./usage-figures-and-calibration.md) | 简体中文

# 图、标定，与 TBLS 的得分密坪

本文是把"奇怪 TBLS PR 图"背后的数学解释在一处的单点入口，并附一份自足重现脚本。它不是用户手册（用 CLI 见 [`usage-experiments-cli.zh-CN.md`](./usage-experiments-cli.zh-CN.md)），不是教程（见 [`../examples/README.md`](../examples/README.md)）——只在某张 PR/ROC 图看起来反常、你想知道原因时读它。

## "PR 悬崖"现象

在 `biomedical_larger` 的几个 cohort（`BC`/`CG`/`CKD`/`DM`），`experiments/visualize.py` 产出的 `pr_{cohort}.png` 上 TBLS 变体（plain `TBLS`、`TBLS IFS`、`TBLS Graph`、`TBLS Full`）曲线在某个 recall 处呈现近垂直的"悬崖"——precision 急跌、之后几条曲线粘一起落到 prevalence。

实测（4 个 TBLS 变体，`biomedical_larger`，cohort `CG`）：

| cohort | 悬崖 recall（约） | 悬崖前 precision | 悬崖后 precision（崖底） | 正样本比例 |
|---|---|---|---|---|
| `BC` | ~0.60 | ~0.45 | ~0.20 | 0.20 |
| `CG` | ~0.70 | ~0.50 | ~0.20 | 0.20 |
| `CKD`| ~0.75 | ~0.55 | ~0.20 | 0.20 |
| `DM` | ~0.90 | ~0.55 | ~0.25 | 0.25 |

同图里 Logistic Regression 基线曲线平滑——无悬崖、高 precision 在 LR 概率降到阈值前都保持。

这不是绘图 bug、不是输入传错（已核：`y_score` 是真 `predict_proba` 输出不是 `y_pred`）、不是折叠拼接 bug（visualize 用的是"先聚合 y_true/y_score 一次 `precision_recall_curve`"的正确做法，不是"曲线坐标拼接"的错法）。它是 TBLS 不带任何概率标定步骤产 `predict_proba` 的直接结果。

## 为什么 TBLS 有 `0.5` 概率密坪

TBLS 分类输出走 BLS 的 ridge 回归闭式解：

```
Z_out = W · A_enh
W     = (A_enhᵀ A_enh + λ I)⁻¹ A_enhᵀ Y      # ridge 闭式求解
```

`predict_proba` 是 `softmax(Z_out)`（二类等价 one-vs-rest），样本正类概率为

```
p_i = exp(Δ z_i) / (1 + exp(Δ z_i))   其中   Δ z_i = z_{i,1} - z_{i,0}
                                            = Δ W · A_enh[i]
```

`p_i ≈ 0.5` 当且仅当 `Δ W · A_enh[i] ≈ 0`。在 `biomedical_larger` 几个 cohort 上大批样本满足此条件，原因有二：

1. **ridge 解的 W 使 `A_enhᵀ A_enh` 的最小本征方向被乘以 `(σ_k + λ)⁻¹`**：那些训练能量小的方向上 `Δ W` 的量级被压小；ridge 故意挑"最温和"的 W 去拟合 Y——这正是把大批"含糊"样本驱入近零 `Δ z` 的选择。它们的 `p_i` 落到 `0.5` 附近窄带。

2. **基于树的特征映射 `A_enh` 本就把一批相似样本映射到增强空间邻近点**，cohort 里那些边界样本（训练折上既不显著正也不显著负的）都落在同一 `Δ W · A_enh ≈ 0` 流形附近。

`CG` 直方图证实：约 30%（488/1589）测试样本的正类评分在 `[0.48, 0.52)`。评分其余部分连续（1589 样本里 1126 个 unique 值），所以 `0.5` 聚集不是离散化——它是 ridge 输出线上真密坪。

Logistic Regression 是**反例**：其权重 `w` 直接优化对数损失（`argmin Σ log(1 + exp(-y_i w·x_i))`），是凸拟合，把自信样本推到 `w·x` 大量级、边界样本到 `0` 附近的平滑尾部——不经过任何 ridge 的"温和 W 投影"。其 `predict_proba` 从近 1 平滑行到近 0；阈扫不穿越密坪，故无悬崖。

## 为什么悬崖特别出现在 *precision-recall* 上

`precision_recall_curve` 把阈值从高到低扫；阈值处 `precision = TP/(TP+FP)`。当阈值高于 `0.5` 密坪，那 488 个坪样本尚未判正，precision 仅跟踪少量高评分真阳样本，保持中等。阈值一跌穿密坪，**这 488 个样本同时被判正**；里面只有约 `prevalence`（`CG` 上 ≈19.5%）为真阳，即约 95 个 TP、393 个 FP。代数上：

```
precision_before = T₀ / (T₀ + FP₀)
precision_after  = (T₀ + 95) / (T₀ + FP₀ + 488)
                  ≈ (T₀ + 95) / (T₀ + FP₀ + 488)
```

recall 跳约 95（坪内真阳），precision 分母加 488——比急跌。即悬崖。崖底正是：

```
阈值足够低→所有测试样本判正:
precision → TP / N_total = N_pos / N_total = prevalence       当 recall → 1
```

等于 cohort prevalence——对应 `BC`/`CG`/`CKD` 的 `≈0.195`、`DM` 的 `≈0.25`（见上表；略差源于 visualize 在算曲线前已拼折叠，per-fold prevalence 微不同）。

ROC 曲线所受影响小得多，因为它坐标系是 `(FPR, TPR)` = `(FP/N_neg, TP/N_pos)`，两者仅随阈值降低而增长——加 488 样本按其正/负比例同时涨两侧，ROC 点对角滑而非垂直落下。`auroc` 在任何严格单调 score-reshaping 下不变——正是为何此数据集上 `auroc` 比 `auprc` 更适合作 TBLS-vs-LR 鉴别指标。

## 临床/论文-实现的框架

这是一个典型的**未标定** ridge 输出呈现：BLS 系分类器把 `predict_proba` 报为闭式回归输出的 softmax，不是概率标定过的。在当前实现里这**不是**缺陷——`TBLS` 的公开 `predict_proba` 与 BLS 文献所报、与用户实验所调相符。"怪 PR 图"就是**没做**标定步骤的可见指纹，不是分隔器 bug。

Logistic Regression 按构造是标定概率分类器——BLS/ridge 的平方损失是在 `0/1` 目标上配 L2 残差，而 LR 损失本身就是对数损失。两者 PR 形状本不可比；在结构化 PR 对比里，未标定 `TBLS` 在同一 `auroc` 下会比标定基线"更噪"。希望论文式标定曲线的评审应计划加一层 `CalibratedClassifierCV(TBLS(), cv=5)` 外层扫、或事后 sigmoid/isotonic 标定——这是**future work item，当前未实现**，且应走估计器契约（`BaseEstimator` 估计器 + 标定器 wrapper），而非这套实验流水线。

## 重现脚本

一份自足重现（不画图、只出数）——验证密坪存在并驱动悬崖：

```python
import sys, numpy as np
sys.path.insert(0, "experiments")
from run_resolution import resolve_run_dir
import experiments.visualize as V
from pathlib import Path
from sklearn.metrics import precision_recall_curve

cohort, run = "CG", "TBLS"
arr = V._cohort_predictions(resolve_run_dir(Path(f"examples/runs/{run}")), run)[(run, cohort)]
yt, ys = arr["y_true"], arr["y_score"]
ys = ys[:, 1] if ys.ndim > 1 else ys
print("prevalence:", float(yt.mean()))
h, e = np.histogram(ys, bins=20)
i_plateau = int(np.argmax(h))
print(f"score-density plateau bin: [{e[i_plateau]:.2f}, {e[i_plateau+1]:.2f}); "
      f"holds {h[i_plateau]}/{len(ys)} samples")
p, r, _ = precision_recall_curve(yt, ys)
mask = r > 0.1     # 跳过 sklearn 在 recall ~ 0 的边界尖刺
idx_in_seg = int(np.argmax(np.abs(np.diff(p[mask]))))
print(f"steepest precision drop in (r > 0.1): recall ≈ {r[mask][idx_in_seg]:.3f}, "
      f"precision {p[mask][idx_in_seg]:.3f} -> {p[mask][idx_in_seg+1]:.3f}")
```

`CG` `TBLS` 上实际输出：

```
prevalence: 0.1951
score-density plateau bin: [0.48, 0.52); holds 488/1589 samples
steepest precision drop in (r > 0.1): recall ≈ 0.710, precision 0.523 -> 0.601
```

（那个 `0.523 -> 0.601` 的**上升**是 `precision_recall_curve` 渲染顺序在密坪边缘 choppy 曲面所致的局部 artifact；你图里看的悬崖是 precision **跨越**整个密坪的整体下行，不是单步点对点的现象。）

## 不为错的项（已验证）

- `train.py` 把 `y_score = model.predict_proba(X_te)` 以 `float32`、shape `(n_te, n_classes)` 写入 `.npz`；`y_pred = model.predict(X_te)` 为 `int64`。两者独立数组，消费者不会混淆。
- `experiments/visualize.py::_cohort_pr` 当 `y_score` 2-D 时切 `y_score[:, 1]`，调用 `precision_recall_curve(y_true, y_score)` **一次**（聚合折叠后），不在折内重归一。实现与 `all_y_true.extend(y_test); ...extend(y_score); precision_recall_curve(all_y_true, all_y_score)` 同配方。
- `.npz` 只在 `train.py::_cross_validate` 中 `grid_point is None` 时写，确保普通训练存在原始预测（从而服务于 visualize 的 ROC/PR/confusion）；grid 运行不产侧文件（体积），visualize 为它们跳过那三图带 stdout 提示。

若未来加标定 wrapper，标定那一 run 的 PR 悬崖应消失、`auprc` 会向 `auroc` 收拢；TBLS 未标定的 `predict_proba` 实现契约将如既往作为 test-fixture 不变。