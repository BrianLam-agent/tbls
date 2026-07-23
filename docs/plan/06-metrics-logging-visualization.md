# Plan 06: Expanded metrics, structured logging (JSONL), and visualization CLI

> Status: final, ready to hand off. **Hard predecessor: Plan 05 (code
> hygiene) should land first** if run close together, since this plan
> touches the same files (`evaluate.py`, `train.py`) — running Plan 05 after
> this one would just re-review already-clean code, which is fine but
> wasteful; running it concurrently risks merge conflicts. Not a strict
> correctness dependency, just a sequencing preference.

## Goal

Three related upgrades to `experiments/`:

1. **Metrics**: fix a real multiclass gap in `TBLSEvaluator`, add a small set
   of additional metrics, refactor `evaluate.py` for clarity.
2. **Logging**: dual-sink logging (human-readable stdout + structured JSONL
   file) with a `TypedDict`-defined event schema, replacing today's plain
   `logging.basicConfig`, so a later analysis/plot pass can consume the JSONL
   directly instead of re-parsing Excel files or stdout text.
3. **Visualization**: a new `experiments/visualize.py` CLI that reads one or
   more experiment output directories' JSONL logs and produces ROC/PR curves,
   a confusion-matrix heatmap, per-fold metric plots, and grid-search summary
   plots — with support for overlaying multiple experiment directories on one
   figure (e.g. comparing ablation variants or grid-search runs).

## Why (context)

Current `experiments/evaluate.py::TBLSEvaluator.calculate_metrics` assumes
**binary** classification: `confusion_matrix(y_true, y_pred).ravel()` unpacks
exactly 4 values (`tn, fp, fn, tp`), which raises for 3+ classes. `TBLS`
itself supports multiclass (`tests/test_tbls.py::test_tbls_multiclass`), so
this is a real, not hypothetical, gap for any dataset with more than 2
classes. Additional single-number summary metrics useful for this project's
imbalanced-data focus (Matthews Correlation Coefficient, Cohen's Kappa) are
currently missing.

Logging today is `logging.basicConfig(level=logging.INFO, format="%(levelname)s
%(message)s")` — one flat stdout stream, string-formatted, nothing persisted
to disk in a form a script can parse back out. The user wants: (a) a
human-readable stream on stdout (keep this), (b) a separate, structured,
line-delimited JSON file next to each run's output so a future analysis pass
doesn't need to scrape strings or reopen every Excel file, and (c) that
structure defined by `TypedDict`s so the event shape is checked by mypy and
discoverable by IDEs, not an ad-hoc dict.

There is currently no visualization at all (`grep` confirms no
`matplotlib`/`seaborn`/plotting code anywhere in `experiments/`), despite the
Excel output already containing everything needed for ROC/PR curves
(`avg_auroc`/`avg_auprc`/per-fold `accuracy` etc.) — nothing renders it.

## Design references

- `experiments/evaluate.py` (current) — the metrics being extended; read the
  whole file before starting (it's already fully quoted in this plan's Step 1
  below, but re-read the live file in case Plan 05 changed it non-substantively).
- `experiments/train.py::_cross_validate`/`_run_grid` — where log events are
  emitted from.
- `docs/usage-experiments-cli.md` — gets updated with the new logging/
  visualization sections; do not duplicate `docs/usage-multiview-fusion.md`'s
  content when discussing multi-view runs' events.

## Non-goals

- Changing the existing Excel output format/columns (additive only — new
  metric columns may be *added*, no existing column renamed or removed, so
  any existing downstream consumer of the Excel files keeps working).
- A web dashboard or interactive plotting (matplotlib static images only, per
  the user's ask — "各种乱七八糟的图片" implies static image files, not a
  live app).
- Retroactively converting old Excel-only run outputs into the new JSONL
  format — the JSONL log is new, produced going forward only.
- Sweeping `CCA_GRID`/`GFCCA_GRID` in `--grid` for multi-view cohorts — that
  remains Plan 02's documented scope limit; Step 7 below is a small,
  optional, separately-gated addition if the user wants it now (see Step 7).

## Implementation steps

### Step 1 — Metrics: fix multiclass, add MCC/Kappa, refactor

In `experiments/evaluate.py`:

- Add a `MetricsDict` `TypedDict` (in a new `experiments/metrics_schema.py`,
  shared with the logging schema in Step 3) documenting every key
  `calculate_metrics` can return — use `total=False` since probability-based
  keys are conditional on `y_score` being given.
- Detect `n_classes = len(np.unique(y_true))` at the top of
  `calculate_metrics`. If `n_classes == 2`, keep **exactly** today's binary
  metric set and values (regression-tested against current behavior — no
  value may change for existing binary-classification callers). If
  `n_classes > 2`:
  - `precision`/`recall`/`f1_score` computed with `average="macro"` (add
    `average="weighted"` variants too, as `precision_weighted`/
    `recall_weighted`/`f1_weighted`, since imbalanced multiclass data cares
    about both).
  - `specificity`/`negative_predictive_value`/`balanced_accuracy`/`gmean`
    computed via `sklearn.metrics.multilabel_confusion_matrix` (one-vs-rest
    per class), then macro-averaged. Use
    `sklearn.metrics.balanced_accuracy_score` directly for
    `balanced_accuracy` rather than hand-deriving it (already correct and
    multiclass-aware).
  - `auroc` via `roc_auc_score(y_true, y_score, multi_class="ovr",
    average="macro")` (requires `y_score` shape `(n, n_classes)`; skip with
    the same `try/except` + warning pattern as today if it fails or
    `y_score` isn't `(n, n_classes)`).
  - `auprc`/`optimal_threshold` are binary-specific (a single ROC/PR curve
    concept) — omit for multiclass rather than forcing a meaningless
    single-curve reduction; document why in the docstring.
- Add for **both** binary and multiclass: `mcc` (`matthews_corrcoef` — works
  for both), `cohen_kappa` (`cohen_kappa_score` — works for both). Add for
  binary only, when `y_score` given: `log_loss`, `brier_score` (
  `brier_score_loss`).
- Split the function body into `_binary_metrics`, `_multiclass_metrics`,
  `_probability_metrics` private helpers, with `calculate_metrics` as the
  public dispatcher — purely a readability refactor, same public signature
  and same return dict shape for existing (binary) callers.
- `calculate_average_metrics`: unaffected in logic, but its input/output
  types should reference `MetricsDict`.

### Step 2 — `experiments/metrics_schema.py` (new)

```python
class MetricsDict(TypedDict, total=False):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    precision_weighted: float   # multiclass only
    recall_weighted: float      # multiclass only
    f1_weighted: float          # multiclass only
    hamming_loss: float
    specificity: float
    negative_predictive_value: float
    balanced_accuracy: float
    gmean: float
    mcc: float
    cohen_kappa: float
    auroc: float | None
    auprc: float | None          # binary only
    optimal_threshold: float | None  # binary only
    log_loss: float | None       # binary only
    brier_score: float | None    # binary only
```

Import this in `evaluate.py` (do not redefine it there) and in Step 3's
event schemas.

### Step 3 — Logging: dual-sink loguru + TypedDict event schema

Add `loguru` to the `experiments` dependency group in `pyproject.toml`.

`experiments/logging_schema.py` (new):

```python
class RunStartedEvent(TypedDict):
    event: Literal["run_started"]
    dataset: str
    model: str
    fusion_method: str | None
    grid: bool

class FoldCompletedEvent(TypedDict):
    event: Literal["fold_completed"]
    dataset: str
    cohort_key: str
    fold: int
    n_splits: int
    metrics: MetricsDict
    grid_idx: int | None
    grid_params: dict[str, object] | None

class GridSummaryEvent(TypedDict):
    event: Literal["grid_summary"]
    dataset: str
    cohort_key: str
    winner_params: dict[str, object]
    winner_metric: float
    n_grid_points: int

class RunFinishedEvent(TypedDict):
    event: Literal["run_finished"]
    dataset: str
    duration_seconds: float
```

`experiments/logging_setup.py` (new): `configure_logging(output_dir: Path,
dataset: str, timestamp: str) -> None` — removes loguru's default handler,
adds:
- stdout sink: level `INFO`, human-readable format (keep the current
  `"{level} {message}"`-equivalent style so existing eyeballed output doesn't
  change unrecognizably), colorized.
- file sink at `{output_dir}/logs/{dataset}_{timestamp}.jsonl`: level
  `DEBUG`, `serialize=True` (loguru's built-in JSON-lines serialization).

Emit events via `logger.bind(**event_dict).info(event_dict["event"])` at each
call site listed in each `TypedDict` above (`train.py`'s CLI entrypoint for
Run{Started,Finished}, `_cross_validate` per fold for `FoldCompletedEvent`,
`_run_grid` after ranking for `GridSummaryEvent`). Loguru's `serialize=True`
sink writes the whole record including `extra` (which holds the bound event
dict) as one JSON object per line — the visualize CLI (Step 5) reads
`record["extra"]` back out. Do not hand-roll JSON writing; use loguru's
built-in serialization so record metadata (timestamp, level) comes for free.

`train.py`: replace `logging.basicConfig(...)` with a call to
`configure_logging(...)` early in the CLI entrypoint (needs `output_dir`,
`dataset`, and a timestamp — reuse whatever timestamp `TBLSResultSaver`
already generates per run, don't generate a second one that could disagree).

### Step 4 — Metrics event wiring in `train.py`

`_cross_validate` already computes a `MetricsDict`-shaped result per fold;
after computing it, emit a `FoldCompletedEvent`. `_run_grid` emits a
`GridSummaryEvent` after ranking. The CLI entrypoint emits
`RunStartedEvent`/`RunFinishedEvent` (timing via `time.perf_counter()`,
already imported in `train.py`).

### Step 5 — `experiments/visualize.py` (new typer CLI)

```
uv run --group experiments python experiments/visualize.py --dir results_dir/tbls_biomedical_larger/DM/20260724_012345
uv run --group experiments python experiments/visualize.py --dir <run1> --dir <run2> --output-dir plots/comparison
```

- Reads one or more `--dir` paths, each expected to contain
  `logs/*.jsonl` (glob all matching files under the given dir, recursively).
  Parses every line as JSON, filters by `record["extra"]["event"]`,
  reconstructs a `pandas.DataFrame` of `FoldCompletedEvent`s (one row per
  fold) and, if present, `GridSummaryEvent`s.
- Produces (via `matplotlib`, added as a new `experiments` dependency):
  - **ROC curve**: needs raw `y_true`/`y_score` per fold, which
    `FoldCompletedEvent` does NOT carry (it only carries scalar metrics, to
    keep the JSONL small) — for ROC/PR curves specifically, this plan adds an
    **optional** `y_true`/`y_score` array pair to `FoldCompletedEvent` (base64
    or a small `.npz` side-file reference, whichever is simpler to implement
    correctly — pick one and document the choice in the acceptance report;
    it must round-trip through JSON, so raw numpy arrays need an explicit
    encoding decision, not "just json.dumps a numpy array").
  - **Confusion matrix heatmap**: same raw-array dependency as ROC/PR; if the
    array-encoding decision above makes this impractical within this plan's
    scope, it is acceptable to render only the scalar-metric-derived plots
    (bar charts, grid-search summaries) in this pass and note the
    raw-prediction plots (ROC/PR/confusion-matrix) as a follow-up needing a
    slightly larger JSONL payload — **do not silently skip these charts
    without saying so in the report**.
  - **Metric-per-fold plots**: bar or box plot of any numeric column in the
    reconstructed fold DataFrame (`balanced_accuracy`, `mcc`, etc.) across
    folds/cohorts — needs only the scalar metrics already in
    `FoldCompletedEvent`, always in scope.
  - **Grid-search summary plot**: metric vs. each swept hyperparameter (one
    subplot per swept param), from `GridSummaryEvent`/grid fold rows —
    always in scope.
  - When multiple `--dir` values are given: tag each source dir's rows (by
    dataset/model/an inferred short label from the dir path) and overlay/
    facet by that tag on the same figures, rather than producing one figure
    per dir.
- Output: PNG files under `--output-dir` (default: `plots/` next to the
  first `--dir` given), one file per plot type, deterministically named.

### Step 6 — Docs

`docs/usage-experiments-cli.md`: new "Logging and visualization" section —
where the JSONL file lives, the event schema (link to
`experiments/logging_schema.py`'s docstrings rather than re-typing every
field), and the `visualize.py` CLI usage with the multi-dir overlay example.

### Step 7 (optional, small, separately gated) — sweep fusion hyperparameters in `--grid`

If the user confirms this is wanted now: extend `_run_grid` so that, for a
multi-view cohort, the swept grid is the Cartesian product of the model grid
**and** the active fusion method's grid (`CCA_GRID`/`GFCCA_GRID`), removing
the Plan 02 scope limit. This is small and independent of Steps 1-6; do not
block the rest of this plan on it. Explicitly confirm with the user before
implementing (this plan does not assume yes).

## Verification commands

```bash
uv run pytest tests/ -v
uv run ruff check . && uv run ruff format --check .
uv run mypy src/tbls
uv build && uvx twine check dist/*
uv run --group experiments python experiments/train.py --dataset biomedical_larger --n-splits 2
uv run --group experiments python experiments/visualize.py --dir <the run just produced>
```

Manually inspect: the produced `logs/*.jsonl` file is valid JSONL (each line
`json.loads`-able), contains the expected event types, and the produced PNGs
actually render sane-looking curves/bars (not blank/all-zero) against real
data.

## Acceptance checklist

- [ ] Multiclass `calculate_metrics` no longer raises; binary output is
      byte-for-byte unchanged from before this plan (regression test against
      the pre-plan binary metric values on a fixed dataset).
- [ ] `mcc`/`cohen_kappa` present for both binary and multiclass; `log_loss`/
      `brier_score` present for binary with `y_score`.
- [ ] `MetricsDict`/event `TypedDict`s exist and are used as the documented
      return/parameter types (checked by `mypy`, even though `experiments/`
      isn't under strict mypy — at minimum no `Any`-typed metrics dict).
- [ ] JSONL log file produced per run, one event per line, valid JSON,
      correct event shapes.
- [ ] `visualize.py` produces at least the scalar-metric-derived plots
      (per-fold bars, grid-search summary) end-to-end on real data; the
      report states plainly whether ROC/PR/confusion-matrix plots needed the
      larger raw-array JSONL payload and whether that was implemented or
      deferred (per Step 5's explicit call-out).
- [ ] Multi-`--dir` overlay works (at least two real runs compared on one
      figure).
- [ ] Existing Excel output unchanged (additive columns only).
- [ ] Docs updated.

## Suggested commits

1. `feat(experiments): multiclass-aware metrics + MCC/Kappa/log-loss/Brier`
2. `feat(experiments): MetricsDict and event TypedDict schemas`
3. `feat(experiments): loguru dual-sink logging (stdout + JSONL)`
4. `feat(experiments): visualize.py CLI (per-fold, grid-search, multi-dir overlay)`
5. `docs: logging schema and visualize.py usage`
