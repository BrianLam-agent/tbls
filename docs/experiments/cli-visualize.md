English | [简体中文](./cli-visualize.zh-CN.md)

# `experiments/visualize.py` — figure CLI

`visualize.py` reads one or more run directories and writes PNG figures. Run
with `uv run --group experiments`.

## Options

### `--dir DIR` (repeatable; required at least once)
- **What**: a run directory, in either form below (the CLI auto-detects which):
  - run-name layer, e.g. `examples/runs/TBLS Full` — the CLI auto-picks the
    newest `YYYYMMDD_HHMMSS` timestamp subdirectory under it;
  - run-name/timestamp layer, e.g. `examples/runs/TBLS Full/20260724_074140`
    — used directly.
- **Conflict**: anything deeper (`.../<timestamp>/logs`), shallower
  (`examples/runs`), or whose subdirectory is not a `YYYYMMDD_HHMMSS` name
  **errors out with a clear diagnostic** — no shell globbing needed.
- **Spaces**: run names with spaces are fine — quote the path on the shell
  (`--dir "examples/runs/TBLS Full"`); the label preserved in figures keeps
  the space.
- **Multiple runs**: pass `--dir X --dir Y --dir Z` to overlay every run on
  the same figures.

### `--output-dir DIR`
- **What**: where the PNGs are written.
- **Default**: `plots/` next to the first `--dir`.

### `--dpi N`
- **What**: PNG resolution.
- **Default**: `300` (print quality). Use `--dpi 120` for quick previews.

## Figures produced

Under `--output-dir`:

| File | When produced | Contents |
|---|---|---|
| `per_fold_metrics.png` | always | `balanced_accuracy` and `mcc` bar charts per cohort, grouped by run — the headline ablation comparison. |
| `grid_search_summary.png` | only when at least one `--dir` is a `--grid` run | metric vs. each swept axis, one subplot per axis. |
| `roc_<cohort>.png` | non-grid runs only | one ROC file per cohort (`roc_DM.png`, `roc_CKD.png`, ...). Each file overlays every run's ROC for **that cohort** — the meaningful ablation comparison. |
| `pr_<cohort>.png` | non-grid runs only | same pattern, precision-recall curves. |
| `confusion_<run>.png` | non-grid runs only | one confusion-matrix sheet per run (cohorts as sub-plots). |

If a grid run is in the `--dir` list, it reports per-fold metrics normally but
**skips** the ROC/PR/confusion plots (grid runs don't write an `.npz`
predictions side-file) and prints a note — same behavior on stdout.

## Why ROC/PR are split per cohort

The natural ablation question is "does run A beat run B on cohort X?", not
"how does run A's curve compare across cohorts". So each `roc_<cohort>.png`
overlays every run on the same axes, and you get one file per cohort rather
than one mixed cohort legend. See [outputs.md](outputs.md) for the npz layout
that feeds the curves.

## Why TBLS PR curves can have a sharp cliff

The TBLS estimator's `predict_proba` is the ridge closed-form output
softmax-transformed **with no probability-calibration step**, so a
high-density band of low-confidence samples can sit at `p≈0.5`. The threshold
sweep crosses that band in one step, dumping a block of mostly-negative
samples into "predicted positive" simultaneously → precision collapses to
prevalence and recall jumps. Full math + reproducer:
[figures-and-calibration.md](figures-and-calibration.md).

## Typical invocations

```bash
# All-in-one ablation comparison: every example run overlaid on the same figures
uv run --group experiments python experiments/visualize.py \
    --dir "examples/runs/TBLS" \
    --dir "examples/runs/TBLS Full" \
    --dir "examples/runs/Logistic Regression" \
    --output-dir examples/plots --dpi 300

# Just one run (for the scalar-metric bars and npz-based plots)
uv run --group experiments python experiments/visualize.py \
    --dir "examples/runs/TBLS Full"
```

## Common errors

- **`--dir ... has no YYYYMMDD_HHMMSS timestamp subdirectory`** — you gave the
  run-name layer but no timestamp dir exists under it (the run probably
  failed). Run `ls examples/runs/{your run_name}`.
- **`--dir ... has no 'logs/' subdirectory; not a valid run timestep
  directory`** — the path you gave *is* a `YYYYMMDD_HHMMSS` directory but its
  contents aren't a `logs/` run (it's a per-cohort Excel dir or something
  else). Give the run-name layer or the run-name/timestamp layer holding
  `logs/`.