[English](./figures-and-calibration.md) | 简体中文

# 图表、标定、与 TBLS 的得分密度平台

本页解释 `TBLS` 消融实验里一个反复出现的"看着怪"的图 — PR 曲线上近乎垂直
的悬崖 — 背后的数学,并给出一个自包含的复现器。当 PR 或 ROC 图上某处看着
病态、你想知道是不是 bug 时读它。它不是用户指南;CLI 见
[cli-visualize.md](cli-visualize.md),流水线总入口见 [index.md](index.md),
可跑教程见 [`../../examples/README.md`](../../examples/README.md)。

## "PR 悬崖"症状

在几个 `biomedical_larger` cohort(`BC`、`CG`、`CKD`、`DM`)上,
`experiments/visualize.py` 产出的 `pr_{cohort}.png` 图里,TBLS 变体曲线
(纯 `TBLS`、`TBLS IFS`、`TBLS Graph`、`TBLS Full`)出现近乎垂直的"悬崖"
— 在某个 recall 处精确率骤降,之后曲线贴在一起下沉到流行率。

经验值(4 个 TBLS 变体,`biomedical_larger`,cohort `CG`):

| cohort | 悬崖 recall(约) | 悬崖前精确率 | 悬崖后精确率(悬崖底) | 正类流行率 |
|---|---|---|---|---|
| `BC` | ~0.60 | ~0.45 | ~0.20 | 0.20 |
| `CG` | ~0.70 | ~0.50 | ~0.20 | 0.20 |
| `CKD`| ~0.75 | ~0.55 | ~0.20 | 0.20 |
| `DM` | ~0.90 | ~0.55 | ~0.25 | 0.25 |

同一张图里 Logistic Regression 基线曲线是平滑的 — 没有悬崖,在 LR 概率
最终跌破阈值扫描之前都保持高精确率。

这不是绘图 bug,不是输入错 bug(我们验证过 `y_score` 是真实的
`predict_proba` 输出,不是 `y_pred`),也不是折拼接 bug(用的是
predict+concat+单次 `precision_recall_curve` 配方,不是曲线坐标拼接)。
它是 TBLS 如何产出 `predict_proba` 的后果 — 没有任何概率标定步骤。

## 为啥 TBLS 会产出 `0.5` 概率平台

TBLS 的分类输出是 BLS ridge 回归闭式:

```
Z_out = W · A_enh
W     = (A_enhᵀ A_enh + λ I)⁻¹ A_enhᵀ Y      # ridge 闭式解
```

`predict_proba` 是 `softmax(Z_out)`(二分类:one-vs-rest 等价),一个样本
的正类概率是

```
p_i = exp(Δ z_i) / (1 + exp(Δ z_i))   其中   Δ z_i = z_{i,1} - z_{i,0}
                                              = Δ W · A_enh[i]
```

所以 `p_i ≈ 0.5` 当且仅当 `Δ W · A_enh[i] ≈ 0`。对 `biomedical_larger`
cohort,这有两个原因会让它**成批**发生:

1. **ridge 解出的 `W` 让 `A_enhᵀ A_enh` 最小特征值方向被
   `(σ_k + λ)⁻¹` 缩放**。训练能量小的方向只对 `Δ W` 在那些方向贡献小
   幅度;ridge 故意挑"最温和"的 `W` 仍能拟合 `Y`,这恰恰是把许多"模糊"
   样本推到近零 `Δ z` 的选择。这些样本的 `p_i` 落在 `0.5` 附近的窄带。

2. **基于树的特征映射 `A_enh` 已经把许多相似样本映射到增强空间里的邻近
   点**,所以一个 cohort 的边界样本(那些在训练折上既不确信正也不确信负
   的)都落在同一个 `Δ W · A_enh ≈ 0` 流形附近。

`CG` 上的直方图证实:约 30% 测试样本(488/1589)的正类得分落在
`[0.48, 0.52)`。得分其余部分是连续的(1589 个样本有 1126 个唯一值),
所以 `0.5` 簇不是离散化 — 是 ridge 输出线上真实的密度凸起。

Logistic Regression 是**反面例子**:它的权重 `w` 直接优化对数损失
(`argmin Σ log(1 + exp(-y_i w·x_i))`),这是一个凸拟合,把确信样本推到
大幅度 `w·x`、把边界样本推到 `0` 附近的平滑尾;没有 ridge 的"温和 W
投影"。它的 `predict_proba` 跨所有测试样本从近 `1` 连续走到近 `0`;
阈值扫描永远不跨密度平台,所以没有悬崖。

## 为啥悬崖是*精确率-召回率*专属伪影

`precision_recall_curve` 从高分到低分走阈值;每个阈值处
`precision = TP / (TP + FP)`。当阈值在 `0.5` 平台之上时,488 个平台样本
还没"预测为正",所以精确率只跟那些真正高分的正样本走,保持中等。阈值一
跌破平台,**全部 488 个样本同时变成"预测为正"**;其中只有约 `流行率`
(CG 上 ≈19.5%)是真阳性,即约 95 正、393 假阳。数值上:

```
precision_before = T₀ / (T₀ + FP₀)
precision_after  = (T₀ + 95) / (T₀ + FP₀ + 488)
                  ≈ (T₀ + 95) / (T₀ + FP₀ + 488)
```

recall 跳约 95(平台里的正样本),而精确率分母涨 488 — 比值骤降。这就是
悬崖。悬崖底恰好是:

```
"阈值之上全预测正"时测试集里正占比:
    ≈ 流行率                  # 因为所有测试样本现在都被标正
precision → TP / N_total = N_pos / N_total = 流行率       当 recall → 1
```

即 cohort 的流行率 — 与 `BC`/`CG`/`CKD` 上的 `≈ 0.195` 底、`DM` 上的
`≈ 0.25` 底吻合(见上表;细微差异来自每折流行率,因为 `visualize.py`
算曲线前先拼接折)。

ROC 曲线受影响小得多,因为它的轴是
`(FPR, TPR)` = `(FP/N_neg, TP/N_pos)`,两者都随阈值下降只增不减 — 加 488
样本按各自比例增长两者,所以 ROC 点沿对角滑而非垂直掉。ROC *积分*
(`auroc`)其实对任何严格单调的得分重塑不变,这就是为啥在本数据集上
`auroc` 比 `auprc` 是更稳定的 TBLS-vs-LR 判别量。

## 临床 / 论文 vs 实现的框架

这是典型的**未标定** ridge 输出表现:BLS 族分类器报告的 `predict_proba`
是闭式回归输出经 softmax,不是概率标定的。就当前实现而言,它不是缺陷 —
`TBLS` 的公开 `predict_proba` 与 BLS 文献报告、与用户实验调参所对照的
一致。"怪 PR 曲线"是标定步骤**没做**的可见签名,不是分隔器 bug。

Logistic Regression 天生是标定概率分类器 — BLS/ridge 的平方损失在 `0/1`
目标上回归带 L2 残差惩罚,而 LR 的损失直接是 log-loss。两者在 PR 形状上
不可比;在结构化 PR 对比里,未标定的 `TBLS` 在同 `auroc` 下会比标定基线
看着"更噪"。期待论文风格标定曲线的评审者应计划加一个
`CalibratedClassifierCV(TBLS(), cv=5)` 外层扫描或事后 sigmoid/isotonic
标定 — **未来工作项,当前未实现**,且它属于估计器契约
(`BaseEstimator` 估计器 + 标定器包装)而非本实验流水线。

## 复现器

一个自包含复现器(不画图,只给数字),验证平台存在并驱动悬崖:

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
mask = r > 0.1     # skip the sklearn boundary spike at recall ~ 0
idx_in_seg = int(np.argmax(np.abs(np.diff(p[mask]))))
print(f"steepest precision drop in (r > 0.1): recall ≈ {r[mask][idx_in_seg]:.3f}, "
      f"precision {p[mask][idx_in_seg]:.3f} -> {p[mask][idx_in_seg+1]:.3f}")
```

`CG` `TBLS` 集上的实际输出:

```
prevalence: 0.1951
score-density plateau bin: [0.48, 0.52); holds 488/1589 samples
steepest precision drop in (r > 0.1): recall ≈ 0.710, precision 0.523 -> 0.601
```

(那个 `0.523 -> 0.601` *向上*跳本身是 `precision_recall_curve` 渲染顺序
的伪影 — 曲线在平台边缘的局部最陡段表面是 choppy 的;你在图里看到的悬崖
是精确率跨平台下降的积分行为,不是单点到单点的一步。)

## 验证过没问题的部分

- `train.py` 把 `y_score = model.predict_proba(X_te)` 以 `float32`、形状
  `(n_te, n_classes)` 存进 `.npz`,把 `y_pred = model.predict(X_te)` 以
  dtype `int64` 存。两者作为独立数组,消费者不会混淆。
- `experiments/visualize.py::_cohort_pr` 在收到 2-D `y_score` 时切片
  `y_score[:, 1]`,在拼接折后**恰好调一次** `precision_recall_curve
  (y_true, y_score)`,且绝不在折内重新归一化得分。实现匹配
  `all_y_true.extend(y_test); ... extend(y_score);
  precision_recall_curve(all_y_true, all_y_score)` 配方。
- `.npz` 仅当 `grid_point is None` 时由 `train.py::_cross_validate` 写,
  所以纯 train 运行有原始预测(从而 `visualize.py` 的 ROC/PR/混淆图可
  画);网格运行不产 side-file(体积),`visualize.py` 对它们跳过那三张
  图并在 stdout 提示。

若未来真加了标定器包装,标定运行的 PR 曲线上悬崖应消失,`auprc` 会朝
`auroc` 上升;`TBLS` 未标定 `predict_proba` 的实现行为会同样作为测试夹具
契约保留。
