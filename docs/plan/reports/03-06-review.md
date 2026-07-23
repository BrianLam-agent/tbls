# Acceptance report (reviewer): Plans 03-06

- **Reviewer:** Claude (independent review, not the implementing agent).
- **Plans:** `03-worked-examples.md`, `04-ablation-variants-and-vectorization.md`,
  `05-experiments-code-hygiene.md`, `06-metrics-logging-visualization.md`.
- **Implementer reports:** `docs/plan/reports/0{3,4,5,6}-*.md` (all `IMPLEMENTED`).
- **Conclusion:** All four **ACCEPTED**, after two bugs found and fixed during
  this review (`c8fae94`, plus a metrics-visualization path issue below), and
  with one significant pre-existing finding (not caused by any of these four
  plans) flagged for a dedicated follow-up plan.

## Concurrent-execution incident (governance note, not a code defect)

Plans 03 and 04 were executed by two agents concurrently on the shared
`master` working tree (confirmed via `git reflog` in both implementer
reports). This caused: (a) `bd17edb` bundling Plan 04's 4 staged perf files
into a Plan-03-authored commit under a mismatched message, and (b) a
`git reset HEAD~1` by the Plan-03 agent that orphaned Plan-04's docs commit
`47f6c14` (self-recovered as `6288073`). **Verified independently**:
`git diff 47f6c14 6288073` is empty (identical content, nothing lost); `git
log`/`git fsck` show no other anomalies; the full test suite is green at
current `HEAD`. No content was lost and no history needed rewriting, but
this should not recur — **recommend serializing agents on one shared branch
going forward** (or use separate worktrees) rather than relying on
self-recovery a second time.

## Independent verification performed

```
uv run pytest tests/ -q        -> 76 passed (after this review's additions)
uv run ruff check . / format --check .   -> clean
uv run mypy src/tbls            -> clean, 19 source files
uv build && uvx twine check dist/*        -> both PASSED
```

Plus hands-on runs: `examples/01_train_tbls.py`, `examples/02_train_tbls_with_grid_search.py`,
`experiments/train.py` (plain + `--grid`), `experiments/visualize.py` (single
and multi `--dir`), and direct reproduction of the ablation factory /
vectorization claims.

## Plan 05 (experiments hygiene): accepted as-is, no issues found

Clean, mechanical, exactly as specified. `classifiers.py` fully translated,
Google docstrings added, zero behavior change (token-equivalence proof is a
nice touch beyond what the plan asked for). Nothing to add.

## Plan 04 (ablation factory + vectorization): accepted, one pre-existing bug surfaced (not a Plan-04 defect)

- `build_tbls_variant` verified: correct switch mapping, correct validation
  errors, exported from `__all__`.
- Vectorization verified independently: reran the bit-for-bit regression
  tests, confirmed `_graph.py`/`gfcca.py`/`_ifs.py` no longer contain the
  nested/neighbor Python loops, confirmed the pre-vectorization `tbls.py`
  solve path (`_solve_weights`) is untouched (only whitespace + the new
  factory function appended).
- **The `use_if_weights=True` + `graph_gamma>0` combination (i.e. GFTBLS —
  the combination the plan's own "Why" table calls "today's tuned default
  combination") collapses to degenerate all-one-class predictions on real
  data.** Reproduced independently:

  ```
  use_if_weights=False graph_gamma=0.0: acc=0.8904 balanced_acc=0.7279
  use_if_weights=True  graph_gamma=0.0: acc=0.8890 balanced_acc=0.7271
  use_if_weights=False graph_gamma=0.1: acc=0.8877 balanced_acc=0.7264
  use_if_weights=True  graph_gamma=0.1: acc=0.8864 balanced_acc=0.5000  <- degenerate
  ```
  (`biomedical_larger.pkl`, cohort `DM`, held-out split, `n_map_trees=10,
  n_enhance_trees=10`.) This was first surfaced by Plan 03's own worked
  example (see below) and is **not caused by Plan 04's vectorization** — the
  vectorized functions are proven bit-for-bit identical to the pre-existing
  loop versions, and `_solve_weights` (where S and L combine:
  `W = (AᵀSA + λI + γAᵀLA)⁻¹AᵀSY`) was not touched by any of Plans 01-06.
  Likely cause (not yet confirmed): the discriminative graph Laplacian
  `L = Lw - β·Lb` is not guaranteed positive semi-definite (subtracting the
  between-class term can make it indefinite), so `γ·AᵀLA` can fight the
  ridge term's positive-definiteness rather than regularize it, and combined
  with `S`'s sample down-weighting this can degenerate the solve. This is a
  **pre-existing latent defect from Plan 01**, only now exposed by Plan 03
  actually exercising both switches together against real data (Plan 01's
  own review checked raw accuracy only, not balanced accuracy — a miss on my
  part at the time). **Tracked as a required follow-up, not blocking Plans
  03/04's acceptance** (04's factory and vectorization are correct on their
  own terms; the collapse is a property of the underlying math, reproducible
  with or without the factory). See "Required follow-up" below.

## Plan 03 (examples): accepted, surfaced the bug above responsibly

Both example scripts run correctly against real data and produce sane,
non-degenerate output. The implementer caught the GFTBLS collapse itself
while writing the example, correctly worked around it (`use_if_weights=True`
alone, `graph_gamma` left at its default `0.0`) rather than silently using a
degenerate configuration or silently "fixing" the library mid-plan, and
flagged it prominently in the acceptance report. This is exactly the right
behavior for an implementer to take when hitting an out-of-scope bug —
noted with approval, not just accepted by default.

## Plan 06 (metrics/logging/visualization): accepted, one real bug found and fixed during this review

- Multiclass metrics: verified `calculate_metrics` no longer raises for 3+
  classes; binary output confirmed byte-identical to pre-plan values on a
  fixed dataset; `mcc`/`cohen_kappa`/`log_loss`/`brier_score` present as
  documented.
- JSONL logging: verified real run output is valid line-delimited JSON with
  the correct event types and counts.
- `visualize.py`: **found and fixed a path-resolution bug** —
  `_load_predictions` hardcoded `run_dir / "logs" / predictions_file`,
  correct only when `--dir` is passed as *exactly* the timestamped run
  directory (the implementer's own manual test happened to use exactly that
  form, per `docs/usage-experiments-cli.md`'s documented example, so it
  passed). But `_cohort_predictions`/event discovery **recursively** globs
  `**/logs/*.jsonl` — clearly designed to also support a shallower `--dir`
  (e.g. sweeping every historical timestamped run under one dataset
  directory at once, a natural way to use the multi-run overlay feature).
  With a shallower `--dir`, npz loading silently found nothing (swallowed by
  a bare `except FileNotFoundError: continue`) while fold-event parsing kept
  working — so ROC/PR/confusion-matrix plots silently disappeared (with only
  a terse "skipped" note) for a `--dir` value the code otherwise clearly
  intended to support. **Fixed** (`c8fae94`): resolve each npz relative to
  its own discovered jsonl's parent directory, not the top-level `--dir`.
  Added `tests/test_visualize.py` (2 parametrized regression tests at
  different nesting depths). Reran the full CLI at both `--dir` depths after
  the fix — both now produce all 5 plot types correctly.
- Excel-additive-only guarantee, `.gitignore` update for `plots/`, and the
  `.npz` side-file design decision all checked and reasonable.

## Required follow-up (not part of this acceptance, tracked for prioritization)

**GFTBLS (`use_if_weights=True` + `graph_gamma > 0`) numerical collapse on
real data** needs its own dedicated investigation — likely centered on
`_solve_weights`'s combination of `S` and a possibly-indefinite `L`. This
affects the literal "recommended tuned combination" from Plan 01's own
framing, so it's a real correctness gap worth prioritizing, not a cosmetic
issue. Recommend a "Plan 07" once the user confirms priority — I have not
written it yet since root-causing needs a bit more investigation first
(candidate directions: clamp/eigenvalue-floor `L` before combining with `S`,
re-derive `discriminative_beta`'s safe range, or add a fit-time diagnostic
that warns on near-degenerate solutions) and I'd rather scope that plan
accurately than guess.

## Working-tree state

`master`, clean except local `.agents`/`.claude`. All four plans' commits +
this review's two fixes (`c8fae94` for `visualize.py`; note the earlier
`view_groups` fix under Plan 02's review was a separate, already-closed
issue) are on `master`. `docs/plan/execution-graph.md` updated to `ACCEPTED`
for all four nodes.
