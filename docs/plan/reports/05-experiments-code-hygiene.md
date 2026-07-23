# Plan 05 acceptance report — `experiments/` code hygiene

- **Plan:** `docs/plan/05-experiments-code-hygiene.md`
- **Node:** `05-experiments-hygiene`
- **Execution date:** 2026-07-24 (`pi` agent resuming the `ex2` session after Plan 04).
- **Conclusion:** `IMPLEMENTED` — all required work + verification complete and
  committed; pending reviewer acceptance. The implementing agent does not grant
  `ACCEPTED`.

## Baseline, branch, and commits

- Branch: `master` (no worktree, no branch switch, as required).
- Session start of Plan 05 work (HEAD when Plan 05 implementation began,
  after Plan 04 was committed): `6c953ec` (a parallel Plan-03 agent's report
  commit — see "Concurrent-execution note" below).
- Hard-predecessor gate: **none** (Plan 05 declares none). The graph records
  the node as `READY` → `IN_PROGRESS` → `IMPLEMENTED` (this report).

Commits made for Plan 05 this session:

| Hash | Subject | Files |
|------|---------|-------|
| `6b62745` | `docs(experiments): translate classifiers.py to English + Google-style docstrings` | `experiments/classifiers.py` |
| `e401dcb` | `build(experiments): enforce Google-docstring D rules on experiments/` | `.ruff.toml`, `docs/development.md` |
| (this commit) | `docs(plan): Plan 05 acceptance report + execution graph (IMPLEMENTED)` | `docs/plan/reports/05-experiments-code-hygiene.md`, `docs/plan/execution-graph.md` |

## Step 1 — Inventory

`rg -c '[^\x00-\x7F]'` per file across `experiments/*.py`:

| file | non-ASCII lines before | after |
|------|-----------------------:|------:|
| `experiments/classifiers.py` | 187 | 0 |
| `experiments/dataprocess.py` | 0 | 0 |
| `experiments/evaluate.py` | 0 | 0 |
| `experiments/hyperparams.py` | 0 | 0 |
| `experiments/multiview.py` | 0 | 0 |
| `experiments/smoke_run.py` | 0 | 0 |
| `experiments/train.py` | 0 | 0 |

All 187 human-facing non-English occurrences were confined to
`experiments/classifiers.py` (the one file that predates the package refactor —
exactly the file the plan called out). The other six files were already clean
ASCII; they needed only a docstring-style audit (Step 3), which found them
already Google-compliant.

## Step 2 — Translate to English

`experiments/classifiers.py` was translated in full: the module docstring, the
`# ---------- section ----------` headers, the conditional-dependency section
comments, every class docstring, every inline implementation comment, every
`print(...)` training-log message, and every `raise ImportError/ValueError`
message string.

The public factory's two `ValueError` messages were kept byte-for-byte equal in
*formatting*: the original `f"未知的分类器 '{name}'. 可选: ..."` (literal quotes around
`{name}`) and `f"未知的距离类型: {self.dist}"` (plain interpolation) map to
`f"Unknown classifier '{name}'. Options: ..."` and
`f"Unknown distance type: {self.dist}"` — **no `!r` conversion added**, so the
rendered strings change language but not format/representation.

## Step 3 — Google-style docstrings

`ruff check experiments/ --select D` (with the global `D100/D104/D107/D203/D213`
ignores) flagged exactly **41** missing public docstrings, **all in
`classifiers.py`**: 40 `D102` (public method) + 1 `D103` (public function
`xavier_init`), across the three `Balanced*Classifier` wrappers,
`MixOmicsBlockPLSDA`, the torch-branch `GraphConvolution/GCN_E/Classifier_1/VCDN`
`forward` methods, and the `MOGONETClassifier`/`MOFAClassifier`/`DIABLOClassifier`/
`SNFClassifier` public methods. Each got a concise one-line Google-style summary.

`__init__` (`D107`) was intentionally left undocumented per the existing global
ignore — the Google convention documents `__init__` in the **class** docstring,
which every class already has.

Private helpers (`_build_adj`, `_build_nn_encoder`, `_predict_proba_tensor`,
`_fit_muon`, `_fit_mofapy2`, `_transform_new`, and the `_SNFClassifierFixed`
subclass's overrides) keep their existing inline comments; the plan explicitly
limits private-helper docstring expansion to trivial one-liners only.

The other six `experiments/*.py` files raised **zero** D violations when D was
enforced — confirming they were already Google-style. So Step 3 added
docstrings only to `classifiers.py`.

## Step 4 — Ruff docstring-lint check

Before this plan, `.ruff.toml` exempted `experiments/**` from the entire `D`
group:
```toml
"experiments/**" = ["D", "ARG001"]
```
Now that `classifiers.py` is brought up to standard, I removed the `D`
exemption (keeping `ARG001` relaxed for typer-CLI options):
```toml
"experiments/**" = ["ARG001"]
```
This is a **zero-extra-diff** ruleset change: `ruff check .` already passes
clean on `experiments/` with `D` enforced (the 41 fixes above closed every
violation). `docs/development.md` §3 was updated to state the Google-style
docstring convention now also applies to `experiments/` (the type-hint/`mypy`
bar for `experiments/` is unchanged — still the published-package-only hard
gate).

## Non-goals honored

- **No behavior change.** Verified two ways:
  1. **Token equivalence.** Filtering out `STRING` and `COMMENT` tokens, the
     new `classifiers.py` produces a code-token sequence **identical** to the
     original (`7646 == 7646` tokens — no logic/identifier/number/operator
     change). Docstrings are `STRING` tokens, so adding them does not perturb
     the code-token comparison.
  2. **Runtime smoke.** `create_classifier('rf'/'knn')` returns the expected
     sklearn estimators; `create_classifier('tbls', n_map_trees=3, ...)` fits a
     12-sample/2-class toy set and `predict_proba` sums to 1; the
     `create_classifier('bogus')` path raises `ValueError` with the translated
     message `"Unknown classifier 'bogus'. Options: 'rf..."`.
- **No stale-docstring discovery.** No docstring was found describing behavior
  that disagreed with the code; nothing required flagging.
- **No docstring-depth expansion.** Only missing-one-liners were added to
  public APIs; no elaborate documentation was invented for private helpers.

## Verification commands and outcomes

| Command | Exit | Observed |
|---------|------|----------|
| `rg -c '[^\x00-\x7F]'` over `experiments/*.py` | 0 | `0` for every file (was 187 in `classifiers.py` only). |
| `uv run ruff check experiments/` (with `D` now enforced) | 0 | `All checks passed!` |
| `uv run ruff format --check experiments/` | 0 | `7 files already formatted` |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run ruff format --check .` | 0 | `38 files already formatted` |
| `uv run --group experiments pytest tests/ -q` | 0 | `65 passed, 22 warnings in 3.24s` (unchanged from the Plan-04 baseline — zero behavior change) |
| `uv run mypy src/tbls` | 0 | `Success: no issues found in 19 source files` (unaffected scope; sanity check) |
| import smoke (`create_classifier` end-to-end) | 0 | `rf -> RandomForestClassifier`, `knn -> KNeighborsClassifier`, `tbls proba shape (12, 2) sum~1: True`, unknown-name `ValueError` raised. |
| code-token equivalence (`tokenize`, drop STRING/COMMENT) | 0 | `code tokens identical (logic unchanged): True (7646 == 7646)`. |

## Acceptance checklist

- [x] Inventory (Step 1's grep output) included above, with a clear statement of
      what was translated (all 187 non-ASCII lines in `classifiers.py`).
- [x] Every public `experiments/*.py` function/class/method has a Google-style
      docstring (41 added to `classifiers.py`; the other six files were already
      compliant; `D` is now enforced, so regressions will fail `ruff check .`).
- [x] Zero behavior change: full test suite passes unmodified; no
      behavior/docstring mismatch was found, so none to flag.
- [x] `ruff check` / `ruff format --check` clean (and now `D`-enforced on
      `experiments/`).

## Deviations from the plan

1. **Two suggested `classifiers.py` commits merged into one.** The plan suggested
   separate "translate" and "add Google docstrings" commits, but both concerns
   were applied to the same file in one coherent rewrite (and the code-token
   equivalence proof covers the whole file at once). Splitting hunks within one
   file between two commits would add risk without value; I combined them into
   `6b62745` with a subject that names both concerns, and added the ruleset
   change as a separate `build(experiments):` commit (`e401dcb`).
2. **Step 4 ruleset extension was taken** (not skipped). The plan made it
   optional pending a nontriviality check; the check passed (zero extra
   `experiments/` lines needed once the 41 fix-ups landed), so enforcing `D` on
   `experiments/` is now a clean lock-in rather than a deferred note.

## Concurrent-execution note

A parallel Plan-03 agent was active on the same shared `master` branch
throughout. During Plan 05's window it committed `017bf66` (examples README +
root README pointer) and `6c953ec` (its Plan 03 acceptance report, explicitly
*deferring* its execution-graph flip to avoid racing the shared graph file).
Neither swept Plan-05 working-tree files, and no `git reset` recurred during
the Plan-05 window — so Plan 05 suffered no cross-lane incident. (Plan 04 did;
see `04-…md`.) The execution graph's Plan 03 row remains `IN_PROGRESS`; the
Plan 03 agent owns that flip and deferred it deliberately.

## Remaining risks / external actions

- **Reviewer acceptance required** before Plan 06 (which lists Plan 05 as a
  sequencing preference) proceeds. This report sets the node to `IMPLEMENTED`,
  not `ACCEPTED`.
- **`docs/development.zh-CN.md`** was not updated (the plan's scope is
  `experiments/` hygiene; the zh-CN convention mirrors the English source and
  is maintained separately). The one-line convention clarification should be
  ported to the translation in a follow-up translation pass if that file tracks
  this section — flagged for the reviewer/translator, not done here to avoid
  scope creep.

## Working-tree state and preserved unrelated changes

After the Plan-05 commits in this report, the working tree is clean except for
this report + the execution-graph flip (staged/to-be-committed together here).
All prior parallel-agent work (`examples/`, `README.md`, the Plan 03 report) was
already committed by that agent and is preserved. All Plan-05 implementation +
this report are on the current branch (`master`); there is no worktree or
branch merge step left for the user.