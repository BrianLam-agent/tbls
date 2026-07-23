English | [简体中文](./usage-multiview-fusion.zh-CN.md)

# Multi-view data contract and CCA/GFCCA fusion

> **Status: convention only, not yet implemented.** No real multi-view
> dataset exists in this project yet (see
> [`docs/plan/02-multiview-cca-gfcca-fusion-convention.md`](./plan/02-multiview-cca-gfcca-fusion-convention.md)
> for the implementation plan). This document defines the pkl contract and
> configuration schema that any future multi-view dataset — and the pipeline
> code that consumes it — must follow, so that once real data exists (the
> motivating case is fusing several image-derived feature views, e.g. up/
> down/left/right fundus fields, or any other number of named views) it can
> be dropped in without renegotiating the format.
>
> **Reading raw images (or any other raw modality) into feature vectors is
> explicitly out of scope of `tbls`/`experiments/`.** This project consumes
> already-extracted per-view feature matrices; how those vectors were
> produced (a CNN embedding, hand-crafted features, anything else) is the
> caller's responsibility.

## 1. What CCA/GFCCA fusion is, briefly

`tbls.cca.PairwiseKCCA` and `tbls.gfcca.GraphFuzzyKCCA` are **two-view**
kernel CCA estimators: given feature matrices from two views of the *same*
samples, they find projection directions that maximize cross-view
correlation, and project each view into that shared space. `GraphFuzzyKCCA`
additionally uses the class labels to make same-class samples project closer
together and to down-weight noisy/borderline samples. See
[`usage-cca-gfcca.md`](./usage-cca-gfcca.md) for the estimator-level API.

`tbls.cca.build_cca_features`/`tbls.gfcca.build_gfcca_features` extend this
to more than two views by running a `PairwiseKCCA`/`GraphFuzzyKCCA` on every
pair of views and concatenating all pairs' projections into one fused
feature matrix. This is existing, already-shipped, unit-tested library
behavior (`docs/usage-cca-gfcca.md`) — the multi-view pipeline work
(Plan 02) is about *wiring it into `experiments/`*, not changing this math.

The fused feature matrix is fed into `TBLS`/`BroadLearningSystem` exactly
like a normal `X` — fusion is a preprocessing step, after per-view scaling/
feature-selection/resampling and before model training.

## 2. The pkl contract

A dataset pkl (loaded with `joblib.load`) is a dict of **cohort keys** (e.g.
`"DM"`, `"CKD"`, or a single implicit cohort — see
[`experiments/datasets/README.md`](../experiments/datasets/README.md) for how
single-cohort files are handled). Each cohort value is itself a dict that is
**exactly one** of:

- **Single-view** (existing, unchanged): `{"data": X, "target": y}` where `X`
  is `(n_samples, n_features)`.
- **Multi-view** (new): `{"views": {name: X_name, ...}, "target": y}` where
  `views` is a dict — not a list — mapping a **stable view name** (any
  string, e.g. `"up"`, `"down"`, `"view_042"`) to that view's feature matrix
  `(n_samples, n_features_of_that_view)`.

```python
# Single-view cohort (today's format, unchanged):
{"DM": {"data": X, "target": y}}

# Multi-view cohort, 4 named views:
{
    "DM": {
        "views": {
            "up":    X_up,     # (n_samples, f_up)
            "down":  X_down,   # (n_samples, f_down)
            "left":  X_left,   # (n_samples, f_left)
            "right": X_right,  # (n_samples, f_right)
        },
        "target": y,           # (n_samples,)
    }
}

# Multi-view cohort, 108 named views -- same shape, no code change needed:
{
    "some_cohort": {
        "views": {f"view_{i:03d}": X_i for i in range(108)},
        "target": y,
    }
}
```

Rules:

- View names are the **stable identity** of a view: what a name means is up
  to the dataset author (`"up"` / `"down"` / `"view_042"` / anything), but it
  must be consistent across every cohort key that uses it, since config
  (`fusion.view_groups`, Section 4) refers to views **by name**, not by
  position. Dict key order carries no meaning.
- All views for one cohort key must have the same `n_samples` and the same
  row-to-sample correspondence (row *i* in every view is the same underlying
  sample). Per-view **feature counts may differ** across views — there is no
  assumption that `f_up == f_down == ...`.
- A cohort dict must have exactly one of `"data"` or `"views"`. Having both,
  or neither (besides `"target"`), is a hard configuration error, not a
  silently-resolved ambiguity.
- A single pkl file may mix single-view and multi-view cohort keys; each key
  is resolved independently based on which of `"data"`/`"views"` it has.
- The existing rules still apply per view: samples with `target == -1` are
  filtered out, labels are otherwise used as-is.

## 3. Preprocessing order

For a multi-view cohort, per k-fold-CV fold:

1. **Split** every view and `y` by the same fold's train/test row indices
   (views stay row-aligned throughout — nothing before fusion may reorder or
   resample one view differently from another).
2. **Per-view preprocessing**, independently for each view: `StandardScaler`
   + `feature_selection` (same `lasso`/`pca`/`mutual_info` options as
   single-view `experiments/dataprocess.py::DataLoader`, just applied once
   per view instead of once total).
3. **Resampling** (row-duplication/removal only — see restrictions below),
   applied identically (same resulting row selection) across *all* views and
   `y` together, regardless of `fusion.view_groups`.
4. **Fusion**, per `fusion.view_groups` (Section 4) — this is the only step
   that is aware of view grouping.
5. The fused feature matrix is handed to `TBLS`/`BroadLearningSystem`
   exactly like single-view `X` today.

### Resampling restrictions (why, and what's supported)

SMOTE-family resamplers (`smote`, `adasyn`, `border_smote`, `smote_tomek`,
`smote_enn`) **synthesize new feature vectors by interpolating existing
ones** in one feature space. There is no way to synthesize a "new sample"
consistently across two *different* feature spaces without inventing an
interpolation scheme the library doesn't expose (interpolating each view
independently would not correspond to one coherent new sample). They are
therefore **not supported for multi-view data** — requesting one is a
configuration error, not a silent fallback.

Two categories are supported, because they only ever *select or duplicate
existing rows* (never synthesize new feature values), so the same row
decision applies unambiguously to every view:

- **Index-only** (`oversample` → `RandomOverSampler`, `undersample` →
  `RandomUnderSampler`): decide which existing rows to duplicate/drop based
  only on class counts in `y`; never look at feature values at all.
- **Reference-view** (`tomek` → `TomekLinks`): decides which majority-class
  rows to drop based on nearest-neighbor structure in *some* feature space —
  ambiguous across views, so a `preprocess.fusion_reference_view` config key
  names which view's feature space to compute Tomek links against; the
  resulting keep/drop row mask is then applied identically to every view.

## 4. Fusion groups

By default (`fusion.view_groups` omitted), every view present in the cohort
is fused together as one group (today's `build_cca_features`/
`build_gfcca_features` all-pairs behavior, unchanged).

To fuse only some views together, and keep others separate (or unfused),
declare a **partition** of the view names into groups:

```yaml
fusion:
  method: gfcca         # "cca" | "gfcca"; default "gfcca" when the cohort is multi-view
  view_groups:
    - [up, down]         # fused together (all-pairs within this group of 2 -> 1 pair)
    - [left, right]      # fused together, independently of the [up, down] group
```

Rules:

- Every view name present in the cohort's `"views"` dict must appear in
  **exactly one** group. A view missing from every group, or listed in more
  than one group, is a configuration error.
- A group is fused with `build_cca_features`/`build_gfcca_features` using
  only that group's views (all-pairs *within* the group only — a group of
  `[a, b, c]` fuses pairs `(a,b)`, `(a,c)`, `(b,c)`, never mixing with
  another group's views).
- **A group of exactly one view is passthrough**: no CCA/GFCCA is run for
  it; that view's own preprocessed features are used as-is for its
  contribution to the final fused matrix. This is how you express "fuse
  `up`+`down`, but leave `left` and `right` untouched and just concatenated
  in": `view_groups: [[up, down], [left], [right]]`.
- The final feature matrix concatenates every group's output, in
  `view_groups` order (or sorted-view-name order per group if a group's own
  internal ordering ever matters — internal ordering only affects which
  pair indices land where, not the math).
- With `N` total views split into groups of sizes `n_1, ..., n_g`, the total
  number of CCA/GFCCA fits is `sum(C(n_i, 2))`, not `C(N, 2)` — grouping is
  also how you keep the pairwise cost tractable for a large `N` (e.g. 108
  views split into small groups of 2-4 is cheap; 108 views in one ungrouped
  group means `C(108, 2) = 5778` pairwise fits, which is almost certainly
  not what you want — grouping is required in that case, not optional).

## 5. Full config example (4 views)

```yaml
dataset: fundus_multiview
model:
  name: tbls
cv:
  n_splits: 5
  random_state: 0
preprocess:
  feature_selection: lasso
  resampling: undersample     # SMOTE-family rejected for multi-view data; see Section 3
  fusion_reference_view: up   # only consulted if resampling: tomek
fusion:
  method: gfcca
  view_groups:
    - [up, down]
    - [left, right]
output_dir: results_dir
```

## 6. Current status

Validated only against a **synthetic** fixture (an arbitrary column split of
`sklearn.datasets.make_classification` output, explicitly not a real
multi-view dataset) as part of Plan 02's acceptance test. No real multi-view
dataset has been ingested. When one is exported (in the format above, with
however many named views), it can be dropped into
`experiments/datasets/` and pointed at with a `fusion` config block per
Section 5 — no further pipeline changes should be required.
