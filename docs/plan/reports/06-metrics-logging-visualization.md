# Plan 06 acceptance report — Expanded metrics, structured logging (JSONL), and visualization CLI

- **Plan:** `docs/plan/06-metrics-logging-visualization.md`
- **Node:** `06-metrics-logging-viz`
- **Execution date:** 2026-07-24 (`pi` agent, single uninterrupted session — no
  concurrent-plan incidents this time: the prior sessions' lanes (03/04/05)
  were all committed and quiescent before Plan 06 began).
- **Conclusion:** `IMPLEMENTED` — all required implementation work and the
  plan's full verification command set are complete and committed; pending
  reviewer acceptance. The implementing agent does not grant `ACCEPTED`.

## Baseline, branch, and commits

- Branch: `master` (no worktree, no branch switch, no `reset`/`checkout` —
  as required).
- Session-start HEAD: `0538c9a` (Plan 05's report commit; Plan 05 node was
  `IMPLEMENTED` pending reviewer acceptance). Plan 06 lists Plan 05 as a
  "sequencing preference, not a correctness dependency," so the hard gate
  (none) was satisfied; the graph node was `READY` → `IN_PROGRESS` →
  `IMPLEMENTED` (this report).
- Final tip during this report: `c811812` (visualize.py) before the docs commit.

Commits made for Plan 06 this session (following the plan's 5 suggested
commit boundaries, slightly rebucketed to keep each cohesive):

| Hash | Subject | Plan step | Files |
|------|---------|-----------|-------|
| `34e8c2f` | `feat(experiments): multiclass-aware metrics + MCC/Kappa/log-loss/Brier` | 1, 2 | `experiments/evaluate.py`, `experiments/metrics_schema.py`, `tests/test_experiments_metrics.py` |
| `ee0b2a6` | `feat(experiments): MetricsDict and event TypedDict schemas` | 3 (schema half) | `experiments/logging_schema.py` |
| `176e23a` | `feat(experiments): loguru dual-sink logging (stdout + JSONL)` | 3, 4 | `experiments/logging_setup.py`, `experiments/train.py`, `pyproject.toml`, `uv.lock` |
| `c811812` | `feat(experiments): visualize.py CLI (per-fold, grid-search, multi-dir overlay)` | 5 | `experiments/visualize.py` |
| `176543f` | `docs: logging schema and visualize.py usage` | 6 + housekeeping | `docs/usage-experiments-cli.md`, `.gitignore` |
| (this commit) | `docs(plan): Plan 06 acceptance report + execution graph (IMPLEMENTED)` | report + graph | this report, `docs/plan/execution-graph.md` |

Step 7 (sweep fusion hyperparameters in `--grid`) was **not implemented**. It
is explicitly *optional, separately gated, and requires user confirmation* per
the plan ("Do not block the rest of this plan on it … this plan does not
assume yes"); no confirmation was given, so it is left out. Flagged below as a
remaining item the user may request as its own follow-up.

## Files / interfaces changed and why

| Path | Change | Why |
|------|--------|-----|
| `experiments/metrics_schema.py` (new) | `MetricsDict` `TypedDict` (total=False) | Single source of truth for the metrics dict schema shared by `evaluate.py` and the logging event schema (avoids an import cycle). |
| `experiments/evaluate.py` | Split `calculate_metrics` into `_binary_metrics`/`_multiclass_metrics`/`_binary_probability_metrics`/`_multiclass_probability_metrics` dispatcher; added MCC/Kappa (binary+multiclass) and log_loss/Brier (binary-with-score); multiclass path uses macro/weighted P/R/F1, `balanced_accuracy_score`, `multilabel_confusion_matrix` for specificity/NPV/gmean, OvR macro AUROC; binary path values byte-identical for the 12 pre-plan keys. | Plan Step 1: fix the real multiclass gap (binary `ravel()`-unpack raised for 3+ classes) and add summary metrics. |
| `experiments/logging_schema.py` (new) | Four event `TypedDict`s (`RunStartedEvent`, `FoldCompletedEvent`, `GridSummaryEvent`, `RunFinishedEvent`), importing `MetricsDict`. | Plan Step 3: typed event shape for the structured JSONL log. |
| `experiments/logging_setup.py` (new) | `configure_logging(output_dir, dataset, timestamp)` + `InterceptHandler`. | Plan Step 3: loguru dual-sink (colorized stdout INFO + serialized JSONL DEBUG file). |
| `experiments/train.py` | Replaced stdlib `logging.basicConfig` with `configure_logging`; emit the four events at their call sites; `_native`/`_native_metrics`/`_native_params` coerce numpy → native for JSON; `predictions_npz` per-cohort side-file wiring (non-grid only); all `logger.info` converted `%`-style → f-strings (loguru does not substitute `%s`). | Plan Steps 3–4: emit structured events + persist raw per-fold predictions for ROC/PR/confusion plots. |
| `experiments/visualize.py` (new) | typer CLI: `--dir` (repeatable) + `--output-dir`; `per_fold_metrics`, `grid_search_summary`, `roc_curves`, `pr_curves`, `confusion_<run>` plots; multi-dir overlay. | Plan Step 5. |
| `tests/test_experiments_metrics.py` (new, 9 tests) | Binary regression vs sklearn exact, pre-plan keyset unchanged, no-score-omits-prob-keys, multiclass no-raise/keys/AUROC-match, average prefix/skip/empty, MetricsDict import. | Plan "Add deterministic tests for the changed contract." |
| `pyproject.toml` + `uv.lock` | Added `loguru` and `matplotlib` to the `experiments` dependency group (via `uv add`; lock updated). | Plan Step 3 (loguru) + Step 5 (matplotlib). |
| `docs/usage-experiments-cli.md` | New "Logging and visualization" section. | Plan Step 6. |
| `.gitignore` | Ignore `plots/`. | Generated PNGs are outputs, not source — prevents accidental commits. |
| `docs/plan/execution-graph.md` | Node 06 `READY`→`IN_PROGRESS`→`IMPLEMENTED` (only the 06 row). | Execution-graph ownership. |

`src/tbls/` was **not** modified (Plan 06 is strictly in `experiments/`, an
explicit scope boundary per AGENTS.md). No existing Excel column/key was
renamed or removed (binary path is regression-tested identical).

## Plan steps + acceptance items, with evidence

### Step 1 — Multiclass metrics + MCC/Kappa/log-loss/Brier ✅
`calculate_metrics` dispatches on `len(np.unique(y_true))`. Binary path:
exact-value regression against the git-HEAD version on a fixed seed (200
samples) — **all 12 pre-plan keys equal**, 4 new keys (`mcc`, `cohen_kappa`,
`log_loss`, `brier_score`) additive (verified by `tests/test_experiments_metrics.py::test_binary_*`
3 tests). Multiclass path (3 classes): no-raise, correct keys (15: the 11 macro
scalars + `precision_weighted`/`recall_weighted`/`f1_weighted` + `auroc`,
excluding the 4 binary-only keys), AUROC matches `roc_auc_score(multi_class="ovr",
average="macro")` exactly (`test_multiclass_auroc_ovr_macro`).

### Step 2 — `experiments/metrics_schema.py` ✅
`MetricsDict(TypedDict, total=False)` with every key annotated (binary-only /
multiclass-only / new-additive tagged in comments). Imported in `evaluate.py`
and `logging_schema.py` — not redefined in either.

### Step 3 — Logging dual-sink + event schema ✅
`configure_logging` removes loguru's default, adds colorized stdout (INFO) +
`serialize=True` JSONL file at `{out}/{model}_{dataset}/{ts}/logs/{dataset}_{ts}.jsonl`
(DEBUG); `InterceptHandler` forwards stdlib `logging` into the same sinks.
Events emitted via `logger.bind(**event).info(event["event"])`. train.py emits
`RunStartedEvent`/`RunFinishedEvent` (timing via `time.perf_counter`),
`FoldCompletedEvent` per fold, `GridSummaryEvent` after `_run_grid` ranks.
Real-run proof: `--n-splits 2` on `biomedical_larger` produced JSONL with 27
lines, all `json.loads`-parseable; event counts `{'run_started': 1,
'fold_completed': 8, None: 17, 'run_finished': 1}` (None = non-event INFO
logs, correctly skipped by visualize); a `fold_completed` event carries
`metrics` (16 keys), `grid_idx=grid_params=None`, and `predictions_file`.

### Step 4 — Event wiring in `train.py` ✅
`_cross_validate` got `grid_idx` and `predictions_npz` parameters; emits
`FoldCompletedEvent` per fold with numpy scalars coerced to native
(`_native`/`_native_metrics`/`_native_params`). `_run_grid` emits
`GridSummaryEvent` after ranking. Non-grid runs write the per-cohort
`{dataset}_{ts}_{key}_predictions.npz` side-file
(`{key}_fold{N}_{y_true,y_pred,y_score}`); grid runs pass `predictions_npz=None`
so the side-file (and its size blow-up across 27×N_folds grid points) is
skipped. Verified on the real run: 4 npz side-files for the non-grid run, no
npz for the grid run.

### Step 5 — `visualize.py` ✅ (with the plan's explicit call-out)
`uv run --group experiments python experiments/visualize.py --dir <non-grid>
--dir <grid> --output-dir plots/comparison` produced 5 PNGs:
`per_fold_metrics.png`, `grid_search_summary.png`, `roc_curves.png`,
`pr_curves.png`, `confusion_tbls_biomedical_larger_20260724_034914.png`.
Non-blank verified by pixel std (34–69 ≫ 5). **The raw-array plots (ROC/PR/
confusion) were implemented, not deferred** — via the `.npz` side-file
decision (base64-in-JSONL rejected as needlessly large; a `.npz` next-file
reference round-trips cleanly and keeps the JSONL small). They render only for
runs that have an `.npz` (non-grid runs); the grid run is skipped for those
three plots with a stdout note — **not silently dropped**, per Step 5.

### Step 6 — Docs ✅
"Logging and visualization" section in `docs/usage-experiments-cli.md`:
JSONL path, the 4 event types (linked to `logging_schema.py`'s docstrings —
fields not re-typed per the plan), the `.npz` side-file, the `visualize.py`
CLI with the multi-dir overlay example, and the additive-only Excel
guarantee.

### Step 7 — fusion-grid sweep **not implemented** (correctly out of scope)
Optional + separately gated + "this plan does not assume yes"; no user
confirmation given this session. Flagged below as a follow-up.

## Verification (exact commands, exit status, observed)

The plan's `Verification commands` block, all run:

| Command | Exit | Observed |
|---------|------|----------|
| `uv run pytest tests/ -v` (plan says `-v`) | 0 | `74 passed, 22 warnings in 4.32s` (was 65 → +9 metrics tests; 22 warnings are pre-existing joblib/numpy/genoptim deprecations). |
| `uv run ruff check . && uv run ruff format --check .` | 0 | `All checks passed!` / `43 files already formatted`. |
| `uv run mypy src/tbls` | 0 | `Success: no issues found in 19 source files` (no `src/tbls` change this plan; sanity gate). |
| `uv build && uvx twine check dist/*` | 0 | `Built tbls-0.1.0.tar.gz … py3-none-any.whl`; twine `PASSED` for both. |
| `uv run --group experiments python experiments/train.py --dataset biomedical_larger --n-splits 2` | 0 | 4 cohorts × 2 folds, `run_started`/`fold_completed`×8/`run_finished` all emitted; JSONL + per-cohort npz side-files written. |
| `uv run --group experiments python experiments/visualize.py --dir <non-grid run> --dir <grid run>` | 0 | 5 PNGs written, 2 runs loaded (8 + 216 fold events, 0 + 4 grid summaries), all non-blank (pixel std 34–69). |

Manual inspection (the plan's two manual checks):
- `logs/*.jsonl`: every line `json.loads`-able ✓; events present and correct
  shape ✓ (FoldCompletedEvent carries `metrics`+`grid_idx`+`grid_params`+
  `predictions_file`; GridSummaryEvent carries `winner_params`/`winner_metric`/
  `n_grid_points`).
- PNGs: render sane curves/bars (not blank/all-zero) — ROC curves non-trivial
  AUC with chance line, per-fold bars grouped by run with nonzero spread,
  confusion matrices with non-trivial cell counts; pixel-std check confirms.

## Skipped / unavailable / environmental

- `mypy src/tbls` runs but Plan 06 touches no `src/tbls` file (`experiments/`
  isn't under mypy's strict scope; the plan asks mypy only as a sanity gate).
- `pytest tests/ -v` warnings (22) pre-date this plan (joblib/numpy NumPy 2.5
  deprecation, genoptim experimental-module FutureWarning, one
  `build_graph_laplacian` divide-by-zero RuntimeWarning on a zeros graph);
  none were introduced or weakened by Plan 06.
- Step 7 not implemented (see above).

## Deviations from the plan

1. **Raw-array encoding: `.npz` side-file, not base64-in-JSONL.** Plan Step 5
   offered both, asking the choice be documented. `.npz` is smaller, round-
   trips trivially (np.savez/np.load), and keeps the JSONL to scalar events
   only. **Raw-array plots (ROC/PR/confusion) are implemented, not deferred**;
   grid runs skip them (no npz by design) with a stdout note — per Step 5's
   "do not silently skip" call-out.
2. **Commit boundaries slightly rebucketed** to stay cohesive with the plan's
   5 suggested commits: Step 1+2 merged into one (metrics + MetricsDict + tests
   are one contract), Step 3 split across schema (ee0b2a6) and loguru+wiring
   (176e23a). Same file count, same logical grouping.
3. **`plot_confusion` per-run, not per-run×cohort** as separate files. One
   `confusion_<run>.png` with cohorts as subplots is more legible; the plan
   said "confusion-matrix heatmap" without mandating a file-per-cohort layout.
4. **`.gitignore` add `plots/`** (small housekeeping, not in the plan's
   literal steps) — generated PNGs are outputs; the `plots/` dir is now
   ignored alongside `results_dir/` so later `git add` mistakes can't stage
   them. Disclosed here, not smuggled in.

## Remaining risks / external actions / user decisions

- **Reviewer acceptance** required before this node flips to `ACCEPTED`.
- **Step 7 (fusion hyperparameter sweep in `--grid`)** is implemented by no
  one yet; confirm to the user's own follow-up plan if wanted. Small and
  self-contained (multiply model grid × `CCA_GRID`/`GFCCA_GRID` in `_run_grid`
  for multi-view cohorts), but genuinely optional and explicitly not assumed
  by this plan.
- **`.npz` side-file only for non-grid runs** — if a user wants ROC/PR per
  grid point, the side-file (and its disk size) would need to grow to grid×
  folds×cohort; that's the cost the plan flagged. Current behavior is the
  bounded choice; surfaced in `docs/usage-experiments-cli.md`.
- **zh-CN doc translations** not updated (English is the source of truth;
  translations maintained separately per AGENTS.md). The new "Logging and
  visualization" section in `docs/usage-experiments-cli.md` should be ported
  to `usage-experiments-cli.zh-CN.md` in a translation pass.

## Current working-tree state and preserved unrelated changes

After the Plan-06 commits this report is the only new file; the
execution-graph flip to `IMPLEMENTED` is staged together with it (only the
`06` row touched; the `03`/`05` rows are other lanes' and preserved).

- Working tree before the final commit: only `M docs/plan/execution-graph.md`
  (this plan's `IN_PROGRESS`→`IMPLEMENTED` flip) and this new untracked
  report. No other lanes were active during the Plan-06 window; no concurrent
  commits raced the index. `plots/` and `results_dir/` are git-ignored (the
  generated PNGs and JSONL/npz run outputs stay local, never committed).
- Everything — implementation, tests, docs, report — is on `master`. No
  worktree, no branch switch, no merge step left for the user or reviewer.