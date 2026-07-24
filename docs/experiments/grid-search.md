# Hyperparameter grid search

`--grid` switches `train.py` from a single k-fold CV to a hyperparameter
sweep. This page tells you what gets swept, how to customize it, and what
lands on disk.

## TL;DR

| Setting | What `--grid` sweeps |
|---|---|
| `model.name: tbls`, no YAML `grid:` | the default `TBLS_GRID` = `n_map_trees: [10,20,40] × n_enhance_trees: [10,20,40] × reg_param: [1e-8,1e-4,1e-2]` = 27 points |
| `model.name: bls`, no YAML `grid:` | `BLS_GRID` = `n_feature_groups: [15,30,60] × n_feature_nodes_per_group: [20,40,80] × reg_param: [0.1,1.0,10.0]` = 27 points |
| `model.name: tbls|bls`, with YAML `grid:` | default ∪ YAML, YAML wins on axis-name collisions |
| `model.name: <baseline>`, with YAML `grid:` | YAML only (baselines have **no** default grid) |
| `model.name: <baseline>`, no YAML `grid:` | **`--grid` ignored**: warns + falls back to a single k-fold run |

## How to use it

Add `--grid` to the `train.py` invocation:

```bash
uv run --group experiments python experiments/train.py \
    --config examples/configs/default.yaml --grid
```

To sweep something else, add a `grid:` block to the YAML. Two examples:

**Example A — sweep only one loop's IFS + graph, pin the rest:**
```yaml
model:
  name: tbls
  use_if_weights: false   # default; overridden per grid point
  graph_gamma: 0.0        # default; overridden per grid point
grid:
  use_if_weights: [false, true]
  graph_gamma: [0.0, 0.05, 0.1]
# n_map_trees / n_enhance_trees / reg_param NOT mentioned -> keep their defaults too
```
That's a 2-axis, 6-point sweep (2 × 3) — the default `TBLS_GRID`'s named axes
that you didn't mention are not swept at all (because none of them appear in
your `grid:`). Wait — that's the catch: if you give `grid:` at all, *every*
default axis you don't name is **kept** at all of its default values during
the sweep. So Example A actually sweeps `use_if_weights × graph_gamma` (6
points) and uses the YAML `model.` defaults for the other `TBLS` hyperparams.

**Example B — sweep a baseline that has no default grid:**
```yaml
model:
  name: lr
grid:
  C: [0.01, 0.1, 1.0, 10.0]
```
That's a 4-point sweep over LR's `C`. Without this YAML `grid:`,
`--grid` for a baseline is silently dropped to a single k-fold run (with a
warning).

## Resolution rule (the merge semantics)

`train.py::_resolve_grid`:

1. **Start from the default** — `TBLS_GRID` for `tbls`, `BLS_GRID` for `bls`,
   nothing for a baseline.
2. **Overlay the YAML `grid:`** on top: every axis named in YAML *replaces*
   the same-named default axis with the YAML's list of values. Every default
   axis you didn't mention is *kept* as-is.
3. Baseline + no YAML `grid:` => explicit error / fallback-to-single-CV,
   never silent.

This gives three common moves in one mechanism:
- **Shrink**: `grid: {reg_param: [1e-8, 1e-4]}` — sweep only that axis, keep the
  others at their (single-default-value) values.
- **Swap**: `grid: {n_map_trees: [5, 50]}` — override the default list for
  that axis.
- **Add**: `grid: {use_if_weights: [false, true]}` — add a brand-new axis
  (the TBLS constructor accepts it as a valid kwarg; swept as a grid point).

## What lands on disk

For every `--grid` cohort:

| Output | Contents |
|---|---|
| Excel `Grid_{i:03d}` sheet | the `_cross_validate` fold rows for grid point `i` |
| Excel `GridSummary` sheet | one row per grid point, ranked by `avg_balanced_accuracy` descending; includes `rank` and `is_winner` columns — **row 1 is the winner** |
| Excel `GridSearchLog` sheet | same contents as `GridSummary` but flat (chronological, not ranked) |
| JSONL `grid_point_completed` event per point | `grid_idx`, `n_grid_points`, `grid_params`, averaged metrics |
| JSONL `grid_summary` event at the end | `winner_params`, `winner_metric`, `n_grid_points` |

(See [outputs.md](outputs.md) for the full event schema and Excel layouts.)

A **`grid_search_summary.png`** figure is produced if you point
`visualize.py` at the `--grid` run; it plots the primary metric vs. each swept
axis as a subplot per axis (see [cli-visualize.md](cli-visualize.md)).

## Scope limits (lifted from Plan specs, still in force)

- **`--grid` does NOT sweep fusion hyperparameters** (`CCA_GRID` /
  `GFCCA_GRID`). Even for multi-view cohorts, it sweeps only the model grid.
  Documented scope limit — not silently dropped. See
  [../usage-multiview-fusion.md](../usage-multiview-fusion.md).
- **Raw `.npz` predictions side-file is not written for `--grid` runs** (size —
  a 27-point × N-fold × N-cohort grid would explode it). ROC/PR/confusion
  plots are skipped for `--grid` runs; the per-fold bar and grid-search
  summary plots still apply. See [cli-visualize.md](cli-visualize.md).