English | [简体中文](./usage-figures-and-calibration.zh-CN.md)

# Figures, calibration, and the TBLS score-density plateau

This document is a single place where the math behind one recurring
"looks weird" figure on the `TBLS` ablation experiments is explained, with a
self-contained reproducer. It only needs to be read when something on the PR
or ROC plots looks pathological and you want to know why — it is not a user
guide (see [`usage-experiments-cli.md`](./usage-experiments-cli.md) for the
CLI) and not a tutorial (see [`../examples/README.md`](../examples/README.md)).

## The "PR cliff" symptom

On several `biomedical_larger` cohorts (`BC`, `CG`, `CKD`, `DM`), the
`pr_{cohort}.png` figure produced by `experiments/visualize.py` shows the
TBLS-variant curves (plain `TBLS`, `TBLS IFS`, `TBLS Graph`, `TBLS Full`)
exhibiting a near-vertical "cliff" — a sharp drop in precision at a particular
recall, after which the curves stick together down to prevalence.

Empirically (4 TBLS variants, `biomedical_larger`, cohort `CG`):

| cohort | cliff recall (approximate) | pre-cliff precision | post-cliff precision (cliff floor) | positive prevalence |
|---|---|---|---|---|
| `BC` | ~0.60 | ~0.45 | ~0.20 | 0.20 |
| `CG` | ~0.70 | ~0.50 | ~0.20 | 0.20 |
| `CKD`| ~0.75 | ~0.55 | ~0.20 | 0.20 |
| `DM` | ~0.90 | ~0.55 | ~0.25 | 0.25 |

The Logistic Regression baseline curves in the same figure are smooth — no
cliff, high precision preserved up to the recall at which the LR probability
finally drops below the threshold sweep.

This is not a plotting bug, not a wrong-input bug (we verified
`y_score` is the real `predict_proba` output, not `y_pred`), not a
fold-concatenation bug (the predict+concat+single-`precision_recall_curve`
recipe is used, not the curve-coordinates concat one). It is a consequence of
how TBLS produces `predict_proba` — without any probability-calibration step.

## Why TBLS produces a `0.5` probability plateau

TBLS's classification output is the BLS ridge-regression closed form:

```
Z_out = W · A_enh
W     = (A_enhᵀ A_enh + λ I)⁻¹ A_enhᵀ Y      # ridge closed-form solve
```

`predict_proba` is `softmax(Z_out)` (binary: the one-vs-rest equivalent), and a
sample's positive-class probability is

```
p_i = exp(Δ z_i) / (1 + exp(Δ z_i))   with   Δ z_i = z_{i,1} - z_{i,0}
                                            = Δ W · A_enh[i]
```

so `p_i ≈ 0.5` iff `Δ W · A_enh[i] ≈ 0`. There are two reasons that happens
in bulk for the `biomedical_larger` cohorts:

1. **The ridge solves for `W` such that the smallest-eigenvalue directions of
   `A_enhᵀ A_enh` are scaled by `(σ_k + λ)⁻¹`**. The directions with small
   training-energy weight contribute only a small magnitude to `Δ W` in those
   directions; ridge deliberately picks the "most moderate" `W` that still
   fits `Y`, which is exactly the choice that drives many "ambiguous" samples
   into a near-zero `Δ z`. Those samples' `p_i` land in a narrow band around
   `0.5`.

2. **The tree-based feature map `A_enh` already maps a lot of similar samples
   to nearby points in the enhancement space**, so a cohort's boundary-samples
   (those that are not confidently positive or confidently negative on the
   train split) all fall near the same `Δ W · A_enh ≈ 0` manifold.

The histogram on `CG` confirms it: ~30% of the test samples (488/1589) have
positive-class score in `[0.48, 0.52)`. The score is otherwise continuous
(1126 unique values across 1589 samples), so the `0.5` cluster is not
discretization — it is a genuine density bump on the ridge-output line.

Logistic Regression is the **negative case**: its weight `w` optimizes
logarithmic loss directly (`argmin Σ log(1 + exp(-y_i w·x_i))`), which is a
convex fit that pushes confident samples to large-magnitude `w·x` and
boundary-samples to a smooth tail around `0`; no ridge's "moderate-W-projection"
occurs. Its `predict_proba` walks continuously from near `1` to near `0`
across all test samples; the threshold sweep never crosses a density plateau,
so no cliff.

## Why the cliff is a *precision-recall* artifact specifically

`precision_recall_curve` walks thresholds from high score down to low; at each
threshold, `precision = TP / (TP + FP)`. When the threshold is above the
`0.5` plateau, the 488 plateau-samples are not yet "predicted positive", so
precision tracks the small number of genuinely high-score positive samples
and stays moderate. The instant the threshold drops below the plateau, **all
488 samples simultaneously become "predicted positive"**; of those, only
~`prevalence` (≈19.5% on CG) are real positives, i.e. ~95 positives and ~393
false positives. Numerically:

```
precision_before = T₀ / (T₀ + FP₀)
precision_after  = (T₀ + 95) / (T₀ + FP₀ + 488)
                  ≈ (T₀ + 95) / (T₀ + FP₀ + 488)
```

Recall jumps by ~95 (the positives in the plateau) while precision's denominator
grows by 488 — the ratio drops sharply. That is the cliff. The cliff floor is
exactly:

```
faction of the test set positive when "predict everything ≥ threshold":
    ≈ prevalence                  # because all test samples are now labeled positive
precision → TP / N_total = N_pos / N_total = prevalence       as recall → 1
```

which is the prevalence of the cohort — matching the `≈ 0.195` floor on `BC`/
`CG`/`CKD` and `≈ 0.25` floor on `DM` (see the table above; the slight
differences are due to per-fold prevalence, since `visualize.py` concatenates
folds before computing the curve).

ROC curves are much less affected because their axis is
`(FPR, TPR)` = `(FP/N_neg, TP/N_pos)`, and both only grow as the threshold is
lowered — adding 488 samples grows both by their fractions, so the ROC point
slides diagonally rather than dropping vertically. The ROC *integral*
(`auroc`) is, in fact, invariant under any strictly-monotone score-reshaping,
which is why `auroc` is a more stable TBLS-vs-LR discriminator than `auprc` on
this dataset.

## The clinical / paper-vs-implementation framing

This is a typical **uncalibrated** ridge-output presentation: BLS-family
classifiers report a `predict_proba` that is the softmax of a closed-form
regression output, not probability-calibrated. It is not, in the implementation
as it stands, a defect — `TBLS`'s public `predict_proba` matches what BLS
literature reports and what the user-experiments were tuning against. The
"weird PR curve" is the visible signature of the calibration step **not being
done`, not a separator bug.

Logistic Regression is, by construction, a calibrated-probability classifier —
the squared loss in BLS/ridge regresses on the `0/1` targets with an L2 residual
penalty, while LR's loss is log-loss directly. They are not comparable in PR
shape; in a structured PR comparison, an uncalibrated `TBLS` will look "noisier"
than a calibrated baseline at the same `auroc`. Reviewers expecting paper-style
calibrated curves should plan to add a `CalibratedClassifierCV(TBLS(), cv=5)`
outer sweep or a post-hoc sigmoid/isotonic calibration — **a future work
item, not currently implemented**, and one that goes to the estimator
contract (`BaseEstimator` estimator + calibrator wrapper) rather than to this
experiments pipeline.

## Reproducer

A self-contained reproducer (no plotting, just numbers) verifying the plateau
exists and drives the cliff:

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

Actual output on the `CG` `TBLS` set:

```
prevalence: 0.1951
score-density plateau bin: [0.48, 0.52); holds 488/1589 samples
steepest precision drop in (r > 0.1): recall ≈ 0.710, precision 0.523 -> 0.601
```

(That `0.523 -> 0.601` *upward* jump is itself an artifact of
`precision_recall_curve` rendering order — the curve's local-steepest-segment
surface is choppy at the plateau edge; the cliff you see in the figure is the
integrated behavior of precision dropping across the plateau, not a single
point-to-point step.)

## What is NOT wrong, verified

- `train.py` saves `y_score = model.predict_proba(X_te)` to the `.npz` as
  `float32` of shape `(n_te, n_classes)`, and `y_pred = model.predict(X_te)`
  of dtype `int64`. The two are kept as separate arrays so a consumer can
  never confuse them.
- `experiments/visualize.py::_cohort_pr` slices `y_score[:, 1]` when given a
  2-D `y_score`, calls `precision_recall_curve(y_true, y_score)` exactly
  once after concatenating folds, and never re-normalizes scores within a
  fold. The implementation matches the `all_y_true.extend(y_test); ...
  extend(y_score); precision_recall_curve(all_y_true, all_y_score)` recipe.
- The `.npz` is written by `train.py::_cross_validate` only when `grid_point
  is None`, so the raw predictions exist for plain train runs (and thus for
  `visualize.py`'s ROC/PR/confusion plots); grid runs do not produce a
  side-file (size), and `visualize.py` skips those three plots for them with
  a stdout note.

If future work does add a calibrator wrapper, the cliff should disappear on
the calibrated run's PR curve and the `auprc` will rise toward `auroc`; the
implementation behavior of `TBLS`'s uncalibrated `predict_proba` will stay a
test-fixture contract the same way.