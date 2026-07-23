# Acceptance report: Plan 01 - Package refactor and real-dataset verification

- **Plan:** `docs/plan/01-package-refactor-and-real-dataset-verification.md`
- **Execution date:** 2026-07-23
- **Branch:** `master`
- **Baseline:** zero commits (the repo had no git history at start; the
  `gitStatus` snapshot showed every file untracked).
- **Implementation commits:** `a94c64d` → `d664ab6` (11 commits, Conventional
  Commits, on `master`).
- **Conclusion:** **IMPLEMENTED** — all implementation work and required
  verification complete and committed; pending reviewer acceptance. Not
  `ACCEPTED` (acceptance requires the reviewer process defined by `AGENTS.md`).

## Summary

The loose-file repo was refactored into an installable `tbls` package
(`src/tbls/`, sklearn-compatible `TBLS`/`BroadLearningSystem`/`PairwiseKCCA`/
`GraphFuzzyKCCA`) with `experiments/` training code outside the package, release
hygiene (ruff, mypy strict, CI, PyPI metadata), and — the actual acceptance bar
— a working end-to-end run against the real `experiments/datasets/*.pkl`:

```
TBLS smoke check OK | key=DM acc=0.9208 macro_f1=0.8985 bls_acc=0.8387 train=1362 test=341 features=204
```

## Implementation commits

| Hash | Subject |
|------|---------|
| `a94c64d` | chore: add .gitignore |
| `8a2c7b4` | chore: add project scaffolding, config, and design docs |
| `79b097e` | build: add pyproject.toml for src-layout, dependency groups, mypy |
| `72514cb` | feat(tbls): add shared _kernel/_ifs/_graph modules |
| `c1f9668` | feat(tbls): migrate bls/tbls/cca/gfcca into src/tbls package |
| `c69d316` | feat(tbls): migrate genoptim and ensemble subpackages (experimental) |
| `bf7fcbc` | feat(tbls): add package __init__, py.typed, and public API |
| `46d808d` | docs: add README |
| `b82219e` | feat(experiments): move real datasets, add CLI, evaluate, smoke_run |
| `bfeeffa` | test: add package test suite and real-dataset smoke test |
| `d664ab6` | chore: remove stale ruff excludes, delete legacy root modules and othercode/ |

## Files / interfaces changed

- **`src/tbls/`** (new package, matches design §3):
  - `_kernel.py`, `_ifs.py`, `_graph.py` — extracted/deduplicated shared logic.
  - `bls.py`, `tbls.py`, `cca.py`, `gfcca.py` — migrated from `othercode/`;
    `tbls.py` refactored to call the shared modules; `PairwiseKCCA`/
    `GraphFuzzyKCCA` gained `BaseEstimator` only (no `TransformerMixin`, per
    §15.2); `get_params`/`set_params` inherited from `BaseEstimator`.
  - `__init__.py` (public API + `__version__`), `py.typed`.
  - `ensemble/` (standalone, portable) and `genoptim/` (encoding/operators
    portable; `fitness.py`/`ga_optimizer.py` import-path-fixed only — their
    TBLS-coupled functions are NOT claimed functional, §15.3). Both emit a
    `FutureWarning` on import.
- **`experiments/`** (new, not on PyPI): `dataprocess.py`, `evaluate.py`
  (`TBLSEvaluator` + `TBLSResultSaver` extracted from legacy root `tbls.py`),
  `train.py` (typer CLI), `classifiers.py` (migrated from `othercode/`),
  `smoke_run.py` (`run_smoke_check`), `configs/default.yaml`,
  `datasets/README.md`, and the two moved `.pkl` datasets.
- **`tests/`** (new): `conftest.py` + 7 `test_*.py`.
- **Config:** `pyproject.toml` (src-layout, dependency groups, mypy strict,
  pytest), `.ruff.toml` (google-convention D rules; D107 ignored; scoped
  per-file-ignores for `experiments/classifiers.py`), `README.md`, `LICENSE`
  (pre-work), `.github/workflows/{ci,release}.yml` (pre-work), `uv.lock`.
- **Deleted (legacy):** root `tbls.py`, `bls.py`, `common.py`, `main.py`,
  `dataprocess.py`, `dataset/mm.py` + empty `dataset/`, `othercode/`,
  root `genetic_optimizer/`, root `ensemble/`. None were ever committed
  (untracked at start), so deletions are not in git history.

## Plan-step evidence

- **Step 0 (git hygiene):** `a94c64d` commits `.gitignore` first; `git log`
  shows it as the root commit.
- **Step 1 (skeleton):** `src/tbls/{genoptim/operators,ensemble}/`,
  `experiments/{configs,datasets}/`, `tests/` created.
- **Step 2 (shared modules):** `_kernel.py`/`_ifs.py`/`_graph.py` written
  (`72514cb`); imported by `tbls.py`/`cca.py`/`gfcca.py`; the duplicated
  `rbf_kernel`/`_compute_if_scores` bodies no longer exist in the migrated
  modules.
- **Step 3 (core algorithms):** bls/tbls/cca/gfcca migrated (`c1f9668`);
  `PairwiseKCCA`/`GraphFuzzyKCCA` inherit `BaseEstimator` only — verified by
  `clone()` round-trip tests.
- **Step 4 (experimental):** genoptim + ensemble migrated (`c69d316`);
  `FutureWarning` verified by `test_genoptim_import_warns` and the import-time
  warnings in the pytest summary. `genoptim.fitness`/`ga_optimizer` carry the
  gap comment and are not tested end-to-end.
- **Step 5 (__init__, py.typed):** `bf7fcbc`; `uv run python -c "import tbls;
  print(tbls.__version__, tbls.__all__)"` → `0.1.0 [...]`.
- **Step 6 (datasets + experiments):** `b82219e`. `.pkl` files **moved** (not
  copied, not deleted) with unchanged byte sizes — `26510612` and `334936122`;
  `dataset/` removed.
- **Step 7 (pyproject):** `79b097e`.
- **Step 8 (tests):** `bfeeffa`; 23 tests pass.
- **Step 9 (delete legacy):** `d664ab6`; no root `.py` modules remain.
- **Step 10 (README/LICENSE):** `46d808d`; README has no TODO; LICENSE is MIT
  with placeholder holder `TBLS Project Contributors` (per §15.1, user replaces
  before release).
- **Step 11 (verify):** see table below.

## Verification commands and observed output

Environment: Windows 11, `uv 0.9.21`, CPython 3.13 (`.python-version`),
numpy 2.5.1, scipy 1.18.0, scikit-learn 1.9.0.

| # | Command | Result |
|---|---------|--------|
| 1 | `uv sync` | exit 0 (base deps numpy/scipy/scikit-learn) |
| 2 | `uv sync --group dev --group experiments` | exit 0 |
| 3 | `uv run python -c "from tbls import TBLS, BroadLearningSystem, PairwiseKCCA, GraphFuzzyKCCA"` | `import OK` |
| 4 | `uv run python -c "... cross_val_score(TBLS(n_map_trees=5, n_enhance_trees=5, random_state=0), X, y, cv=3)"` | `[0.91176471 0.81818182 0.84848485]` |
| 5 | `uv run pytest tests/ -v` | `23 passed, 6 warnings in 1.44s` |
| 6 | `uv run ruff check .` | `All checks passed!` |
| 7 | `uv run ruff format --check .` | `32 files already formatted` |
| 8 | `uv run mypy src/tbls` | `Success: no issues found in 19 source files` |
| 9 | `uv build` | built `dist/tbls-0.1.0-py3-none-any.whl` + `dist/tbls-0.1.0.tar.gz` |
| 10 | `uvx twine check dist/*` | `PASSED` for both artifacts |
| 11a | `uv run --group experiments python experiments/smoke_run.py` (manual) | `TBLS smoke check OK \| key=DM acc=0.9208 macro_f1=0.8985 bls_acc=0.8387 train=1362 test=341 features=204` |
| 11b | `uv run pytest tests/test_real_dataset_smoke.py -v` (manual, data present) | `1 passed, 4 warnings in 1.05s` (ran, not skipped) |

The built wheel contains **only** `tbls/` (21 package files + `py.typed` +
dist-info) — no `experiments/`. The sdist includes `src/tbls` + `tests`.

The 6 pytest warnings are: the expected `FutureWarning` from importing the
experimental `tbls.ensemble`/`tbls.genoptim` subpackages, and a
`DeprecationWarning: Setting the shape on a NumPy array has been deprecated in
NumPy 2.5` emitted by **joblib** while unpickling the legacy `.pkl` (not by
`tbls` code) — environmental, from the dataset file format.

## Deviations from the plan (and why)

1. **`mypy python_version` bumped 3.10 → 3.13** (design §11 / pyproject).
   numpy 2.5.1's bundled stubs use PEP 695 `type` statements that mypy cannot
   parse under a 3.10 target. 3.13 is the dev interpreter; the package's
   `requires-python = ">=3.10"` runtime floor is still enforced by the CI test
   matrix (3.10–3.13). Documented in a comment in `pyproject.toml`.
2. **`D107` added to `.ruff.toml` ignore list.** `select = ["D"]` re-enabled
   D107 (missing `__init__` docstring) contrary to the declared
   `convention = "google"`, which documents `__init__` params in the class
   docstring. Ignoring D107 makes the config self-consistent with its stated
   google convention.
3. **scipy/sklearn are untyped in this env** (scipy 1.18.0 and sklearn 1.9.0
   ship no `py.typed`). Per design §11, targeted `# type: ignore[import-untyped]`
   was added at each scipy/sklearn import line in `src/tbls`. Resulting
   `Any`-propagation was handled with explicit `NDArray` annotations on the
   scipy/sklearn return values (e.g. `sq_dists: NDArray[np.float64] = cdist(...)`)
   and `# type: ignore[misc]` on the four classes that subclass the untyped
   `BaseEstimator`/`ClassifierMixin`. All justified, no global strict loosening.
4. **`experiments/classifiers.py` per-file-ignores** (`ARG002`, `N801`, `N812`,
   `E402`, `F401`) in `.ruff.toml`. This 1390-line legacy comparison-algorithm
   factory (not part of the published package, not exercised by the smoke run)
   has rules that flag external-interface conformations that must not be changed
   in working code (sklearn `get_params(deep=...)`, PyTorch `import ... as F`,
   public class names, availability-check imports). All *mechanical* violations
   in it (C408, B905, B007, F841, E722, E721, SIM108, RET504, RUF001 Chinese
   strings translated to English) were fixed.
5. **`genoptim.fitness`/`ga_optimizer.py`: `model` typed as `TBLS` with targeted
   `# type: ignore`** on the intentionally-broken coupled calls
   (`predict(trees=...)`, `mapping_trees`, decode int-vs-bool), per §15.3
   ("leave bodies as-is"). The `from tbls import TreeBroadLearningSystem` line
   was replaced with `from tbls.tbls import TBLS` (import-path fix) and the
   annotation updated; bodies are otherwise verbatim.
6. **`_increment_layer` dead params removed.** The original signature took
   `Y_onehot, S, L` but never used them (ARG002). Removed the params and the
   corresponding caller args — no behavior change (they were unused).
7. **`.ruff.toml` legacy `exclude` entries removed** (`othercode`,
   `genetic_optimizer`, `ensemble`, `bls.py`, `tbls.py`, `common.py`, `main.py`,
   `dataprocess.py`, `dataset`). Per the file's own note ("remove these entries
   once the corresponding files are gone"). **Important:** these patterns also
   matched `src/tbls/tbls.py`, `src/tbls/bls.py` and `src/tbls/ensemble/`, so
   they had been silently hidden from ruff until the excludes were removed; the
   newly-exposed issues (UP037, RUF022, RUF059, ARG002) were fixed in `d664ab6`.
8. **`pytest pythonpath = ["."]`** added to `pyproject.toml` so
   `tests/test_real_dataset_smoke.py` can do `from experiments.smoke_run import
   run_smoke_check` (experiments/ is a namespace package, no `__init__.py`).
9. **Stale `.venv` removed.** The pre-existing `.venv` was an incomplete 3.14.2
   venv (only `python.exe` + `tqdm.exe`, no installed packages) and `uv sync`
   could not recreate it ("access denied" removing `.venv\Scripts`). It is a
   git-ignored build artifact; removing it let `uv sync` create a clean venv.

## Remaining risks / external actions / user decisions

- **PyPI release is out of scope** (plan assumption A3). Before publishing: fill
  in `pyproject.toml` `authors` and `[project.urls]` (currently `TBD`), confirm
  the `tbls` name on PyPI (have a fallback ready), replace the LICENSE
  `TBLS Project Contributors` placeholder if personal attribution is wanted
  (§15.1), push to GitHub, and configure the PyPI Trusted Publisher
  (§8.1/§12.2). The `release.yml` workflow and `twine check`-clean build are
  already in place.
- **`genoptim.fitness`/`ga_optimizer` are not functional** against the current
  `TBLS` (§15.3) — by design. A follow-up plan would decide whether `TBLS`
  grows a subtree-prediction API.
- **joblib `DeprecationWarning`** when loading the legacy `.pkl` (numpy 2.5
  array-shape deprecation) — environmental, from the dataset file format, not
  from `tbls`. No action required unless the dataset is re-serialized.
- **Cython kernels** are out of scope (later phase); the candidate hot paths are
  marked in the shared module docstrings.

## Working-tree state

- Branch `master`, 11 commits, clean except for untracked agent-tooling
  directories (`.agents/`, `.claude/`, `CLAUDE.md`) which are **preserved
  unrelated work** and were intentionally not committed (not part of the plan
  scope; the `.gitignore` ignores only `.claude/settings.local*.json`).
- `experiments/datasets/biomedical_larger.pkl` (26510612 B) and
  `data_cross_train.pkl` (334936122 B) exist on disk, git-ignored.
- `dist/` build artifacts were removed after verification (git-ignored).
- The work is entirely on `master`; **no worktree merge or branch switch is
  needed**.
