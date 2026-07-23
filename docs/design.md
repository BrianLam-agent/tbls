# TBLS Package Refactor Design

> Status: **Reviewed — amended, ready for execution**
> Last updated: 2026-07-23 (review pass by pi)

This document is the single source of truth for the TBLS package refactor. It
is reviewed and amended before any execution. Once frozen, the "Execution
Plan" section becomes the actionable task list.

---

## 0. Review Addendum (read this first)

This section records the issues found in the original draft and how they were
resolved. The rest of the document already reflects the fixes — this is a
changelog for the review, not a duplicate spec.

### 0.1 Critical fix: the dataset must not be deleted

The original draft's file-migration table (§4.3) listed `dataset/*.pkl` under
**"Deleted"** with the reason "large data files, gitignored". This conflated
two different things: **git-ignoring** a file (keep it on disk, don't track
it in version control) and **deleting** it. `dataset/biomedical_larger.pkl`
(26 MB) and `dataset/data_cross_train.pkl` (335 MB) are the *only* real data
in this repository and the whole point of the refactor is to run the new
package against them. If an execution agent had followed §4.3 literally with
`rm -rf`, the user would have lost their dataset.

**Fix**: §4.1/§4.2/§4.3 below now say explicitly: *move* the two `.pkl` files
to `experiments/datasets/` (kept on local disk), and only `.gitignore` them.
Nothing under `experiments/datasets/*.pkl` is ever deleted by the plan.

### 0.2 Git hygiene ordering (repo has zero commits)

`git log` shows this repository has **no commits yet** — every file,
including `.gitignore` itself, is currently untracked. This is actually a
good moment to get this right: if `.gitignore` is committed *first*, the
335 MB dataset file, `.venv/`, `__pycache__/`, and `.mypy_cache/` never enter
git history at all, which avoids the classic mistake of committing large
binaries and then needing `git filter-repo`/BFG to remove them later. §13
(Execution Plan) now has an explicit step 0 for this.

### 0.3 Pre-work done now instead of left to the execution agent

The user asked whether some tasks can be done ahead of time rather than
handed to the execution agent. The following were low-risk, didn't depend on
the file-tree migration, and are **already applied to the repo** as of this
review (not just described in this doc):

- **`.gitignore`** — rewritten (was minimal: only pycache/venv/build). Now
  also covers `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, coverage
  files, IDE folders, OS cruft, experiment result outputs, and — importantly
  — the dataset files under both `dataset/` (current) and
  `experiments/datasets/` (post-migration), so the transition doesn't
  accidentally commit them either.
- **`.ruff.toml`** — the previous file was **a leftover from an unrelated
  project** (its `src = ["ptcg_env", "dmc", "cli", "tests"]` and
  `known-first-party = ["ptcg_env", "dmc", "cli"]` reference module names
  that don't exist anywhere in this repo — this was copy-pasted from
  elsewhere and never adapted). It has been replaced with a config for
  *this* project: `src/`, `experiments/`, `tests/`, `target-version = "py310"`
  to match the package's `requires-python`, and — the important addition —
  the `D` (pydocstyle) rule set with `convention = "google"`. The original
  draft's §11 mandated Google docstrings as a binding coding convention but
  its own proposed `ruff.toml` didn't select `D`, so the convention would
  never actually have been enforced by CI. Fixed.
- **`.pylintrc`** — deleted. It was a 23 KB untouched pylint default dump,
  not referenced by `.pre-commit-config.yaml` or `pyproject.toml`, and fully
  superseded by ruff (which the `.pre-commit-config.yaml` already uses as
  the sole linter). Keeping two linter configs around, one of them dead
  weight, invites confusion about which is authoritative.
- **`LICENSE`** — added (MIT, matching the assumed license in §8). Copyright
  holder is placeholder `TBLS Project Contributors`; **replace with your
  name/org before the first PyPI release** if you want personal attribution
  (see §15.1).
- **`.github/workflows/ci.yml`** and **`.github/workflows/release.yml`** —
  added. See §12 (new). Neither the original draft nor the repo had any CI
  at all — for a project whose explicit goal is "publish to PyPI and make it
  a proper project," this is not optional. The workflows reference
  `src/tbls` and `tests/` paths that don't exist yet; they'll simply start
  passing once the execution plan's migration steps land. They don't run
  yet either way since there is no git remote/GitHub repo configured.

### 0.4 Other gaps found and fixed in the sections below

- No type-checking story at all, despite the code already being fully type
  hinted and a stale `.mypy_cache/` proving mypy had been run manually before.
  → Added §11 (mypy config + `py.typed` marker + CI job).
  → **Note**: `.mypy_cache/` itself should be deleted before the first commit
  (already covered by the new `.gitignore`, and it's a cache directory with no
  historical value).
- No CI/CD at all. → Added §12.
- No PyPI-readiness checklist (classifiers, project URLs, trusted publishing,
  name availability). → Added §8.1 and §12.
- `experiments/requirements.txt` in the original draft contradicts the fact
  this project already uses `uv` (there's a committed `uv.lock`). A second,
  disconnected requirements file is exactly the kind of drift `uv` is meant
  to prevent. → Replaced with a `uv` dependency group (`[dependency-groups]
  experiments = [...]`) in the root `pyproject.toml`; see §8.
- Test plan (`tests/test_bls.py`, `test_tbls.py`, `test_cca.py`) didn't cover
  `gfcca`, `genoptim`, `ensemble`, or — critically — an end-to-end run against
  the *real* dataset, which is the user's actual acceptance bar ("跑通这个
  dataset 的数据"). → Expanded in §9 (renumbered from the original tests
  section) with a `conftest.py`, per-module smoke tests, and a mandatory
  real-data smoke script + verification step.
- `PairwiseKCCA` / `GraphFuzzyKCCA` transformer question was left open. →
  **Correction after re-review**: the original amendment said "adopt
  `TransformerMixin` now" — this was checked against the actual
  `othercode/cca.py` / `othercode/gfcca.py` source and turned out to be wrong.
  Both classes are inherently **two-view** estimators: `fit(X1, X2[, y])`,
  `transform_view1(X_new)`, `transform_view2(X_new)`, and a no-argument
  `transform()` that returns the *training* projections of both views (used
  by `build_cca_features`). None of this matches sklearn's single-argument
  `TransformerMixin.transform(X) -> array` contract, and inventing a fake
  single-`X` `transform` would either silently drop the second view or
  require a non-standard tuple convention that breaks `Pipeline`/`FeatureUnion`
  anyway. Resolved in §15.2 (revised): keep the existing multi-view method
  names unchanged; add `BaseEstimator` only (for `get_params`/`set_params`/
  `clone()` consistency), not `TransformerMixin`. Full `Pipeline` compatibility
  for these two classes is explicitly out of scope for this refactor.
- `genoptim`/`ensemble` experimental-status question was left open. →
  Resolved in §15.3, **with a scope correction**: `ensemble/diversity_metrics.py`
  and `ensemble/tree_selector.py` are standalone (no TBLS coupling) and port
  directly. `genetic_optimizer/fitness.py` and `ga_optimizer.py`, however, call
  `model.predict(X, trees=selected_trees)`, read `model.mapping_trees`,
  `tree.selected_features`, `model.tree_params["bootstrap_ratio"]`,
  `model.n_map_nodes`, and `model.X_original` — none of which exist on the new
  `tbls.tbls.TBLS` (which has `predict(X)` with no `trees` kwarg, `map_trees_`,
  and `RegressionTreeModule.feature_indices_`). This is a genuine API gap, not
  a renaming exercise, and closing it (e.g. adding per-tree-subset prediction
  to `TBLS`) is out of scope for a refactor plan — it's new functionality.
  Fix: `genoptim` is migrated as a **directory move + import path fix only**;
  the `TBLS`-coupled functions keep their old logic but the plan does **not**
  claim they run successfully against the new `TBLS`, and no test asserts
  `GeneticOptimizer.optimize()` succeeds end-to-end. The `FutureWarning`
  (still added) explicitly says the module is coupled to internals that may
  not exist. This keeps the plan honest and executable instead of silently
  shipping a subtly broken "adaptation."
- Python version floor left open. → Resolved in §15.4: `>=3.10`, CI matrix
  tests 3.10–3.13 (dev machine is on 3.13 per `.python-version`).
- `pyproject.toml` in the original draft was missing PyPI metadata
  (classifiers, `[project.urls]`, keywords) that reviewers/`twine check`
  expect from a "proper" package. → Added in §8.1.
- No versioning strategy beyond a hardcoded `version = "0.1.0"`. → Kept
  static (simplicity wins for a first release) but documented the bump
  procedure and recommended `hatch-vcs` as a future improvement, not required
  now (§8.2).

---

## 1. Context & Goals

The current repository is a collection of loose files with two parallel
implementations of every algorithm: a root-level legacy version (hand-written
CART trees, bugs in incremental update and `predict_proba`, no sklearn
compatibility) and an `othercode/` rewrite (sklearn-compatible, correct,
paper-faithful). There is no package layout, no `__init__.py`, and (before
this review) no working `.gitignore`/lint config/build configuration.

### Goals

1. Produce a `pip install`-able `tbls` package publishable to PyPI.
2. Algorithm code lives **inside** the `tbls` package; training/CLI/data
   pipeline code lives **outside** it, in `experiments/`.
3. Identify and prepare hot computational kernels for Cython acceleration
   (Cython implementation is a *later* phase; this refactor lays the
   structure and isolates the kernels so they can be swapped in).
4. Establish release hygiene: `.gitignore`, `ruff`, `mypy`, CI/CD, complete
   `pyproject.toml`.
5. Verify the refactored package reproduces working end-to-end training runs
   against the real datasets in `dataset/` (moved to `experiments/datasets/`)
   on the user's machine — this is the actual acceptance bar, not just
   "imports without error."

### Non-goals (this phase)

- Implementing the Cython kernels themselves.
- Writing full grid-search experiment harnesses (next phase; the package is
  structured to support them).
- Matching the original paper's reported metrics exactly (that's a modeling
  question, not a refactor question) — the bar is "the pipeline runs
  end-to-end and produces sane, non-NaN metrics," not "beats the paper."

---

## 2. Decisions

| Question | Decision |
|----------|----------|
| Which TBLS/BLS version is canonical | `othercode/` versions (sklearn-compatible). Legacy root versions are deleted. |
| Genetic optimizer & ensemble modules | Move into package as `tbls.genoptim` and `tbls.ensemble`, marked experimental (docstring + `FutureWarning` on import). `ensemble/*` is standalone and fully adapted; `genoptim/*` is a directory move + import-path fix only — its `TBLS`-coupled functions are **not** verified against the new `TBLS` API (real gap, see §0.4/§15.3). No separate PyPI extra needed — they add no new dependencies. |
| `othercode/classifiers.py` mega-factory | Move entirely to `experiments/`; not part of the published package. |
| C/C++ acceleration path | Cython + pyproject build. This phase isolates kernels; implementation is later. |
| External scripts layout | `experiments/` directory alongside the package, as a `uv` dependency group, not a separate `requirements.txt`. |
| CLI | In `experiments/` only; not a packaged `[cli]` extra. YAML config + typer CLI overrides. |
| Config format | YAML + typer. |
| Subpackage naming | `genetic_optimizer` → **`genoptim`** (no underscores). |
| `PairwiseKCCA` / `GraphFuzzyKCCA` sklearn role | Keep the existing two-view `fit(X1, X2[, y])` / `transform_view1` / `transform_view2` / `transform()` API unchanged; add `BaseEstimator` only, not `TransformerMixin` (§15.2 — signature mismatch with sklearn's single-`X` transform contract). |
| Type checking | `mypy` in strict-ish mode, `py.typed` marker shipped in the wheel. |
| CI/CD | GitHub Actions: lint + typecheck + test matrix (3.10–3.13) + build on every push/PR; publish-on-release via PyPI Trusted Publisher (OIDC, no stored token). |
| Dataset files | Real `.pkl` files are **moved**, never deleted, to `experiments/datasets/`; git-ignored, kept on local disk. |

### Coding conventions (binding for all code in this refactor)

1. **Docstrings**: Google style, written in English, on every public class and
   function. Enforced by `ruff` (`D` rules, `convention = "google"` — see the
   already-updated `.ruff.toml`), not just a stated convention.
2. **Comments**: English.
3. **Type hints**: Every public function/method signature fully typed;
   `mypy` must pass on `src/tbls` in CI.
4. **ML models**: Every machine-learning model in the package **must** be a
   sklearn-compatible estimator (`BaseEstimator` + `ClassifierMixin` or
   `RegressorMixin`), implementing `fit` / `predict` / `predict_proba` /
   `get_params` / `set_params`, so `GridSearchCV` and `cross_val_score` work
   in `experiments/`. Feature extractors (`PairwiseKCCA`, `GraphFuzzyKCCA`)
   additionally inherit `BaseEstimator` for `get_params`/`set_params`/`clone()`
   consistency, but keep their existing two-view method names
   (`fit(X1, X2[, y])`, `transform_view1`, `transform_view2`, `transform()`)
   unchanged — see §15.2 for why `TransformerMixin` does not fit this API.

---

## 3. Target Directory Structure

```
tbls/                                  # repo root
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .ruff.toml
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
│
├── docs/
│   └── design.md                      # this file
│
├── src/                               # PyPI package source
│   └── tbls/
│       ├── __init__.py                # public API
│       ├── py.typed                   # PEP 561 marker (empty file)
│       ├── _kernel.py                 # shared kernel utilities
│       ├── _ifs.py                    # shared IFS score computation
│       ├── _graph.py                  # shared graph Laplacian construction
│       ├── bls.py                     # BroadLearningSystem (sklearn estimator)
│       ├── tbls.py                    # TBLS (sklearn estimator)
│       ├── cca.py                     # PairwiseKCCA (transformer) + feature builders
│       ├── gfcca.py                   # GraphFuzzyKCCA (transformer) + feature builders
│       │
│       ├── genoptim/                  # [experimental] genetic optimizer
│       │   ├── __init__.py            # emits FutureWarning on import
│       │   ├── encoding.py
│       │   ├── fitness.py
│       │   ├── ga_optimizer.py
│       │   └── operators/
│       │       ├── __init__.py
│       │       ├── selection.py
│       │       ├── crossover.py
│       │       └── mutation.py
│       │
│       └── ensemble/                  # [experimental] tree selection
│           ├── __init__.py            # emits FutureWarning on import
│           ├── diversity_metrics.py
│           └── tree_selector.py
│
├── experiments/                       # training scripts + CLI (NOT on PyPI)
│   ├── configs/
│   │   └── default.yaml
│   ├── classifiers.py                 # comparison-algorithm factory
│   ├── dataprocess.py                 # data loading / preprocessing
│   ├── evaluate.py                    # metrics + result saving
│   ├── train.py                       # typer CLI: YAML config + overrides
│   ├── smoke_run.py                   # minimal real-dataset smoke test (§9.3)
│   └── datasets/
│       ├── README.md                  # explains data provenance; files gitignored
│       ├── biomedical_larger.pkl      # moved from dataset/, gitignored, NOT deleted
│       └── data_cross_train.pkl       # moved from dataset/, gitignored, NOT deleted
│
└── tests/
    ├── conftest.py                    # shared synthetic-data fixtures
    ├── test_bls.py
    ├── test_tbls.py
    ├── test_cca.py
    ├── test_gfcca.py
    ├── test_genoptim.py
    ├── test_ensemble.py
    └── test_real_dataset_smoke.py     # skipped if experiments/datasets/*.pkl absent
```

### Rationale for `src/` layout

The `src/tbls/` layout prevents accidental imports of the in-repo package
without installation (forces testing against the installed package, catches
packaging bugs early). It is the recommended modern Python layout.

---

## 4. File Migration Map

### 4.1 Into `src/tbls/` (package core)

| Source | Destination | Transformation |
|--------|-------------|----------------|
| `othercode/tbls.py` | `src/tbls/tbls.py` | Move; refactor `_compute_kernel_matrix` / `_kernel_distance_matrix` → `_kernel`; `_compute_if_scores` → `_ifs`; `_build_graph_laplacian` → `_graph`. Add Google docstrings + full type hints. |
| `othercode/bls.py` | `src/tbls/bls.py` | Move as-is; add Google docstrings. Already sklearn-compatible. |
| `othercode/cca.py` | `src/tbls/cca.py` | Move; `rbf_kernel` → call `_kernel.rbf_kernel`; `project_cca_features` stays here; `PairwiseKCCA` gains `BaseEstimator` (not `TransformerMixin` — §15.2). |
| `othercode/gfcca.py` | `src/tbls/gfcca.py` | Move; dedupe `rbf_kernel`; `_compute_if_scores` → call `_ifs.compute_if_scores_simple`; `GraphFuzzyKCCA` gains `BaseEstimator` (not `TransformerMixin` — §15.2). |
| `genetic_optimizer/*` | `src/tbls/genoptim/*` | **Move + import-path fix only.** Update `from tbls import TreeBroadLearningSystem` → `from tbls.tbls import TBLS` in `fitness.py`/`ga_optimizer.py` so the module *imports*. Do **not** attempt to reconcile `model.predict(X, trees=...)`, `model.mapping_trees`, `tree.selected_features`, `model.tree_params["bootstrap_ratio"]`, `model.n_map_nodes`, `model.X_original` against the new `TBLS` — these attributes/kwargs do not exist on it and adding them is new functionality, out of scope for this refactor (§0.4/§15.3). `ensemble/*` has no such coupling and is fully portable. Rename dir. Add `FutureWarning` in `__init__.py`. |
| `ensemble/*` | `src/tbls/ensemble/*` | Move; update imports; audit attribute access against new `TBLS` (`map_trees_`, `selected_features`, etc. — confirm these attribute names still exist on the new estimator, add/rename in `tbls.py` if not). Add `FutureWarning` in `__init__.py`. |
| *(new)* | `src/tbls/_kernel.py` | Extracted from cca/gfcca/tbls. |
| *(new)* | `src/tbls/_ifs.py` | Extracted from tbls/gfcca. |
| *(new)* | `src/tbls/_graph.py` | Extracted from tbls. |
| *(new)* | `src/tbls/py.typed` | Empty marker file so downstream type checkers pick up `tbls`'s type hints (PEP 561). Must be added to `[tool.hatch.build.targets.wheel]`. |

### 4.2 Into `experiments/` (not on PyPI)

| Source | Destination | Transformation |
|--------|-------------|----------------|
| `othercode/classifiers.py` | `experiments/classifiers.py` | Move; update imports to `from tbls import ...`. |
| `dataprocess.py` | `experiments/dataprocess.py` | Move as-is. |
| `main.py` | `experiments/train.py` | Rewrite as typer CLI with YAML config; call `tbls` package. |
| legacy `tbls.py` `TBLSEvaluator` | `experiments/evaluate.py` | Extract metrics computation. |
| legacy `tbls.py` `TBLSResultSaver` | `experiments/evaluate.py` | Extract Excel result saving. |
| **`dataset/biomedical_larger.pkl`** | **`experiments/datasets/biomedical_larger.pkl`** | **Move (not copy, not delete). File stays on local disk; only git-ignored.** |
| **`dataset/data_cross_train.pkl`** | **`experiments/datasets/data_cross_train.pkl`** | **Move (not copy, not delete). File stays on local disk; only git-ignored.** |
| *(new)* | `experiments/smoke_run.py` | Minimal script: load one dataset from `experiments/datasets/`, fit `TBLS` on a small split, assert predictions are finite and `predict_proba` rows sum to 1. This is the user's actual "跑通这个 dataset" acceptance check — see §9.3. |

### 4.3 Deleted (actually deleted — code only, never data)

| File | Reason |
|------|--------|
| root `tbls.py` | Legacy, buggy, superseded by `othercode/tbls.py`. |
| root `bls.py` | Legacy, superseded by `othercode/bls.py`. |
| `common.py` | Duplicates `classifiers.py` with a non-sklearn interface. |
| `main.py` (root) | Replaced by `experiments/train.py`. |
| `dataprocess.py` (root) | Moved to `experiments/`. |
| `dataset/mm.py` | One-off inspection script; not needed after migration. |
| `othercode/` | Emptied after migration (all files moved out per §4.1/§4.2). |
| `__pycache__/`, `.mypy_cache/` | Build/cache artifacts, never tracked; safe to delete from disk any time. |

**`dataset/*.pkl` are not in this table.** They are moved per §4.2, not
deleted. `dataset/` itself becomes empty and can be removed once the move is
confirmed, but the two `.pkl` files themselves must exist somewhere on disk
(`experiments/datasets/`) at the end of the migration.

---

## 5. Public API (`src/tbls/__init__.py`)

```python
"""Tree-based Broad Learning System (TBLS) for classification."""

from tbls.bls import BroadLearningSystem
from tbls.cca import (
    PairwiseKCCA,
    build_cca_features,
    project_cca_features,
)
from tbls.gfcca import (
    GraphFuzzyKCCA,
    build_gfcca_features,
)
from tbls.tbls import TBLS

__version__ = "0.1.0"  # kept in sync with pyproject.toml; see §8.2

__all__ = [
    "TBLS",
    "BroadLearningSystem",
    "PairwiseKCCA",
    "build_cca_features",
    "project_cca_features",
    "GraphFuzzyKCCA",
    "build_gfcca_features",
]
```

Underscore-prefixed modules (`_kernel`, `_ifs`, `_graph`) are private and not
exported. `genoptim` and `ensemble` are importable but carry an
`experimental` note in their module docstrings, and emit a `FutureWarning`
the first time either subpackage's `__init__.py` executes, e.g.:

```python
# src/tbls/genoptim/__init__.py
import warnings

warnings.warn(
    "tbls.genoptim is experimental: its API is coupled to TBLS internals "
    "and may change without notice between minor versions.",
    category=FutureWarning,
    stacklevel=2,
)
```

---

## 6. Shared Module Extraction Specification

### 6.1 `src/tbls/_kernel.py`

Consolidates three duplicated implementations:

- `othercode/cca.py::rbf_kernel` and `othercode/gfcca.py::rbf_kernel` (identical).
- `othercode/tbls.py::_compute_kernel_matrix` and `::_kernel_distance_matrix`.

Public functions:

```python
def rbf_kernel(X: np.ndarray, Y: np.ndarray | None = None,
               gamma: float | None = None) -> np.ndarray:
    """Gaussian RBF kernel with adaptive median-heuristic gamma.

    Args:
        X: First sample matrix, shape (n, d).
        Y: Second sample matrix, shape (m, d). If None, Y = X.
        gamma: Base scale. If None, uses median-heuristic.

    Returns:
        Kernel matrix, shape (n, m).
    """

def compute_kernel_matrix(X: np.ndarray) -> np.ndarray:
    """RBF kernel matrix K = rbf_kernel(X) with adaptive gamma."""

def kernel_distance_matrix(K: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances derived from a kernel matrix."""
```

> **Cython kernel candidate #1**: `rbf_kernel`. The
> `cdist(X, Y, 'sqeuclidean')` + `exp(-gamma * D)` is O(n*m*d) and is on the
> hot path of TBLS.fit, CCA.fit, and GFCCA.fit.

### 6.2 `src/tbls/_ifs.py`

Two distinct formulas coexist; both kept, in one module:

- From `othercode/tbls.py::_compute_if_scores` — GEIB formulation (Chen et al.,
  IEEE TFS 2025), returns a diagonal matrix `S`.
- From `othercode/gfcca.py::_compute_if_scores` — simplified
  membership/non-membership/hesitancy, returns a vector `s`.

Public functions:

```python
def compute_if_scores_geib(X, y, K=None, if_sigma=1.0) -> np.ndarray:
    """GEIB IFS scores. Returns diagonal weight matrix S, shape (n, n)."""

def compute_if_scores_simple(A, y, sigma_if=1.0, delta_if=0.5,
                             min_weight=1e-4) -> np.ndarray:
    """Simplified IFS scores. Returns vector s, shape (n,)."""
```

> **Cython kernel candidate #4**: the per-sample neighborhood loop computing
> `Lambda[i] = mean(y[neighbors] != y[i])` is Python-level O(n * k).

### 6.3 `src/tbls/_graph.py`

Extracted from `othercode/tbls.py::_build_graph_laplacian`.

```python
def build_graph_laplacian(X, y, K=None, graph_alpha_in=1.0, graph_alpha_p=1.0,
                          graph_knn=10, use_kernel=True) -> np.ndarray:
    """Combined intrinsic/penalty graph Laplacian L = a_in*L_in - a_p*L_p.

    Uses normalized Laplacians and kernel-space distances.
    """
```

> **Cython kernel candidate #3**: kNN mask + similarity weight construction
> has a Python-level loop over edge pairs.

---

## 7. Sklearn-Estimator Contract

Every ML model in `src/tbls/` must satisfy:

- Inherit `sklearn.base.BaseEstimator` and `ClassifierMixin` (or
  `RegressorMixin` where applicable).
- All constructor args stored as attributes with identical names (sklearn
  clone requirement).
- Implement `fit(X, y) -> self`, `predict(X)`, `predict_proba(X)`.
- Implement `get_params(deep=True)` and `set_params(**params)` (free if
  inheriting BaseEstimator with consistent naming).
- Set `self.classes_`, `self.n_classes_` in `fit`.
- No I/O, no logging to files, no `tqdm` inside the estimator.
- Full type hints on public methods; passes `mypy` (§11).

Feature extractors (`PairwiseKCCA`, `GraphFuzzyKCCA`) additionally:

- Inherit `sklearn.base.TransformerMixin` (+ `BaseEstimator`).
- Implement `fit(X, y=None) -> self` and `transform(X) -> np.ndarray`;
  `fit_transform` comes free from `TransformerMixin`.

Current status:
- `othercode/tbls.py::TBLS` — ✅ already compliant.
- `othercode/bls.py::BroadLearningSystem` — ✅ already compliant.
- `PairwiseKCCA` / `GraphFuzzyKCCA` — currently plain classes; need
  `TransformerMixin` + rename `project_*` call sites to `transform` during
  migration (see §15.2 for rationale).

---

## 8. `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tbls"
version = "0.1.0"
description = "Tree-based Broad Learning System for classification"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "TBD", email = "TBD" }]  # fill in before first release, see §15.1
keywords = ["machine-learning", "broad-learning-system", "decision-trees", "sklearn", "classification"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy",
    "scipy",
    "scikit-learn",
]

[project.urls]
Homepage = "https://github.com/TBD/tbls"      # fill in once the repo is pushed
Repository = "https://github.com/TBD/tbls"
Issues = "https://github.com/TBD/tbls/issues"

[project.optional-dependencies]
cython = ["cython"]      # C/C++ extension build (later phase)

[dependency-groups]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]
experiments = [
    "typer",
    "pyyaml",
    "pandas",
    "imbalanced-learn",
    "xgboost",
    "openpyxl",
    "joblib",
    "tqdm",
]

[tool.hatch.build.targets.wheel]
packages = ["src/tbls"]

[tool.hatch.build.targets.sdist]
include = ["src/tbls", "tests"]

[tool.mypy]
python_version = "3.10"
packages = ["tbls"]
mypy_path = "src"
strict = true
warn_unused_ignores = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

Key change: the published package depends **only** on numpy/scipy/sklearn.
`pandas`, `xgboost`, `imbalanced-learn`, `openpyxl`, `tqdm`, `joblib`,
`typer`, `pyyaml` are `experiments/`-only dependencies, installed via a `uv`
dependency group (`uv sync --group experiments`) rather than a second,
disconnected `requirements.txt` — this repo already has a `uv.lock`, so a
plain `requirements.txt` would immediately drift out of sync with it.

### 8.1 PyPI readiness checklist

Before running `uv publish` / triggering `release.yml` for the first time:

1. Confirm the package name `tbls` is actually available on PyPI (search
   pypi.org — a generic 4-letter name may be taken; have a fallback name,
   e.g. `tbls-ml` or `tree-bls`, ready).
2. Fill in `authors`, `[project.urls]` (real GitHub URL once pushed).
3. Push the repo to GitHub, then configure a PyPI **Trusted Publisher**
   (Settings → Publishing on pypi.org) pointing at
   `<org>/<repo>` + `.github/workflows/release.yml` + environment `pypi`.
   No `PYPI_API_TOKEN` secret is needed with this flow.
4. `uv build && uvx twine check dist/*` locally before the first tagged
   release.

### 8.2 Versioning

Static `version = "0.1.0"` in `pyproject.toml` (mirrored in
`tbls.__version__`) is fine for the first release — don't add
`hatch-vcs`/`setuptools-scm` machinery until there's an actual need for
per-commit dev versions. To cut a release: bump both version strings in the
same commit, tag `vX.Y.Z`, push the tag, create a GitHub Release from it —
`release.yml` (§12) does the rest.

---

## 9. Tests

### 9.1 Layout

```
tests/
├── conftest.py                    # shared synthetic-data fixtures (classification X/y,
│                                   # small n so tests run in milliseconds)
├── test_bls.py                    # BroadLearningSystem: fit/predict/predict_proba,
│                                   # get_params/set_params round-trip, sklearn clone()
├── test_tbls.py                   # TBLS: same contract, plus incremental-layer behavior
├── test_cca.py                    # PairwiseKCCA: fit(X1,X2)/transform_view1/2, BaseEstimator get_params/set_params
├── test_gfcca.py                  # GraphFuzzyKCCA: same
├── test_genoptim.py               # encoding.py + operators/* only (pure functions, no TBLS coupling);
│                                   # does NOT test fitness.py/ga_optimizer.py end-to-end (see §15.3)
├── test_ensemble.py               # tree_selector / diversity_metrics smoke tests
└── test_real_dataset_smoke.py     # see §9.3; skipped (not failed) if data absent
```

### 9.2 Sklearn-compatibility check

`test_bls.py`/`test_tbls.py` must include
`sklearn.utils.estimator_checks.check_estimator` or, if full compliance isn't
realistic (e.g. due to custom `X`/`y` shape assumptions), at minimum a
`clone()` round-trip and a `GridSearchCV`/`cross_val_score` smoke run on
synthetic data, matching the original draft's stated goal in §2.

### 9.3 Real-dataset smoke run (the user's actual acceptance bar)

This is new relative to the original draft, which only verified
"imports work" and "synthetic-data tests pass." The user's stated goal is to
**run this refactored package against the data in `dataset/` on their own
machine.** That must be an explicit, checkable step:

- `experiments/smoke_run.py`: loads one real dataset from
  `experiments/datasets/*.pkl`, does a small train/test split, fits `TBLS`
  (and `BroadLearningSystem`) with modest hyperparameters (small `map_num`
  so it runs in seconds even on 335 MB source data — subsample if needed),
  and asserts: no exceptions, no `NaN`/`Inf` in predictions,
  `predict_proba` rows sum to 1, accuracy is not stuck at a trivial constant
  (e.g., not always predicting the majority class with 0 variance) — enough
  to know the port is *substantively* correct, not just import-clean.
- `tests/test_real_dataset_smoke.py`: same assertions as a pytest test,
  using `pytest.mark.skipif` to skip (not fail) when
  `experiments/datasets/*.pkl` isn't present — so `uv run pytest` still
  passes in CI (where the 335 MB file will never exist) while giving a real
  signal locally, on the user's machine, where it does exist.
- This must be run and shown to pass **manually, on the user's machine**,
  as part of accepting the plan — it cannot be fully automated in CI given
  the dataset size, so don't let "CI is green" substitute for it.

---

## 10. `experiments/train.py` CLI Design

```python
"""TBLS training CLI: YAML config with typer command-line overrides."""

from pathlib import Path
import yaml
import typer

app = typer.Typer()


@app.command()
def train(
    config: Path = typer.Option("configs/default.yaml", help="YAML config path."),
    dataset: str = typer.Option(None, help="Override config dataset name."),
    map_num: int = typer.Option(None, help="Override mapping node count."),
    n_splits: int = typer.Option(None, help="Override CV fold count."),
) -> None:
    """Run a TBLS training experiment from a YAML config."""
    cfg = yaml.safe_load(config.read_text())
    if dataset is not None:
        cfg["dataset"] = dataset
    if map_num is not None:
        cfg["model"]["map_num"] = map_num
    if n_splits is not None:
        cfg["cv"]["n_splits"] = n_splits
    # ... execute pipeline


if __name__ == "__main__":
    app()
```

`configs/default.yaml` sketch:

```yaml
dataset: depression
data_path: experiments/datasets/

model:
  name: tbls
  map_num: 10
  enhance_num: 10
  reg_param: 2.0e-15
  incremental_method: ge_if

preprocess:
  feature_selection: lasso
  resampling: smote

cv:
  n_splits: 5
  random_state: 42

output_dir: results_dir
```

`results_dir/` (and any `experiments/results/`) are git-ignored — see the
updated `.gitignore`.

---

## 11. Type Checking (`mypy`)

New relative to the original draft. Config lives in `pyproject.toml`
(§8, `[tool.mypy]`):

- `strict = true`, `mypy_path = "src"`, scoped to the `tbls` package only
  (not `experiments/`, which is a lower-bar internal tool — type hints
  there are nice-to-have, not enforced).
- Ship `src/tbls/py.typed` (empty file) so `tbls` is recognized as typed by
  downstream consumers' type checkers (PEP 561). Must appear in
  `[tool.hatch.build.targets.wheel]` packaging (it's inside `src/tbls/`, so
  it's included automatically by the `packages = ["src/tbls"]` glob — just
  don't forget to actually create the file, hatchling won't invent it).
- CI job `typecheck` in `.github/workflows/ci.yml` runs
  `uv run mypy src/tbls` on every push/PR.
- Third-party stubs: `scipy`/`sklearn` have partial or missing stubs in some
  versions; if `mypy --strict` proves too noisy against them, prefer
  targeted `# type: ignore[import-untyped]` at the import line over
  loosening `strict` globally — keep the strict bar for `tbls`'s own code.

---

## 12. CI/CD

New relative to the original draft (which had no CI/CD section at all). Two
workflows, already added to the repo at `.github/workflows/`:

### 12.1 `ci.yml` — on every push/PR

1. **lint**: `uv run ruff check .` + `uv run ruff format --check .`.
2. **typecheck**: `uv run mypy src/tbls`.
3. **test**: matrix over Python 3.10–3.13, `uv run pytest tests/ -v
   --cov=tbls`. (The real-dataset smoke test auto-skips here — §9.3.)
4. **build**: `uv build` + `uvx twine check dist/*`, artifact uploaded for
   inspection. Confirms the wheel is well-formed *before* any release.

### 12.2 `release.yml` — on GitHub Release published

Builds the sdist/wheel and publishes to PyPI via `pypa/gh-action-pypi-publish`
using OIDC **Trusted Publishing** (no long-lived `PYPI_API_TOKEN` secret
needed) — see §8.1 for the one-time PyPI-side setup.

### 12.3 Pre-commit

`.pre-commit-config.yaml` already exists and already runs `ruff --fix` +
`ruff-format` with `--config=./.ruff.toml`, which is correct and needs no
change. Optionally add a local `mypy` hook once `src/tbls` exists — deferred
because pre-commit mypy hooks re-installing the venv per run is slow; running
`mypy` in CI (§12.1) is sufficient for now.

---

## 13. Execution Plan

> This section becomes the actionable checklist once the design is approved.

0. **Git hygiene first** (see §0.2): with `.gitignore`, `.ruff.toml`,
   `LICENSE`, and `.github/workflows/*` already added by this review,
   `git add .gitignore && git commit -m "chore: add .gitignore"` **before**
   staging anything else, so large/generated files never enter history.
   Then proceed with the rest of this plan as normal commits.
1. Create directory skeleton: `src/tbls/`, `src/tbls/genoptim/operators/`, 
   `src/tbls/ensemble/`, `experiments/configs/`, `experiments/datasets/`,
   `tests/`.
2. Write shared modules: `src/tbls/_kernel.py`, `_ifs.py`, `_graph.py`
   (extracted, deduplicated, Google docstrings, full type hints).
3. Migrate core algorithms:
   - `othercode/bls.py` → `src/tbls/bls.py`
   - `othercode/tbls.py` → `src/tbls/tbls.py` (refactored to use shared modules)
   - `othercode/cca.py` → `src/tbls/cca.py` (add `BaseEstimator`; keep two-view API)
   - `othercode/gfcca.py` → `src/tbls/gfcca.py` (add `BaseEstimator`; keep two-view API)
4. Migrate experimental modules:
   - `genetic_optimizer/` → `src/tbls/genoptim/` (rename dir, fix imports
     against the *new* `TBLS` API — audit attribute names, don't assume
     compatibility; add `FutureWarning` on import)
   - `ensemble/` → `src/tbls/ensemble/` (same audit + warning)
5. Write all `__init__.py` files (package + subpackages) and
   `src/tbls/py.typed`.
6. **Move** `dataset/biomedical_larger.pkl` and `dataset/data_cross_train.pkl`
   to `experiments/datasets/` (move, not copy, not delete — §4.2/§4.3).
   Delete `dataset/mm.py` and the now-empty `dataset/` directory.
7. Build `experiments/`: `train.py` (typer CLI), `evaluate.py`,
   `dataprocess.py`, `classifiers.py`, `configs/default.yaml`,
   `datasets/README.md`, `smoke_run.py`.
8. Update `pyproject.toml` to the §8 content (dependency groups, mypy config,
   PyPI metadata — fill in real author/URL placeholders where known).
9. Write tests per §9: `conftest.py` + all `test_*.py` files, including
   `test_real_dataset_smoke.py`.
10. Delete legacy files: root `tbls.py`, `bls.py`, `common.py`, `main.py`,
    `dataprocess.py`, `othercode/` (now empty).
11. Write/update `README.md` (installation, quickstart example fitting
    `TBLS` on synthetic data, link to the paper this implements, note on
    `genoptim`/`ensemble` experimental status).
12. Verify (see §14) — including the manual real-dataset run per §9.3.

---

## 14. Verification

1. `uv sync` succeeds (installs only numpy/scipy/sklearn for the base package).
2. `uv sync --group dev --group experiments` succeeds.
3. `uv run python -c "from tbls import TBLS, BroadLearningSystem, PairwiseKCCA, GraphFuzzyKCCA"` succeeds.
4. `uv run python -c "from sklearn.model_selection import cross_val_score; from tbls import TBLS; ..."` — sklearn compatibility check.
5. `uv run pytest tests/ -v` passes (real-dataset test auto-skipped if data absent).
6. `uv run ruff check .` clean.
7. `uv run ruff format --check .` clean.
8. `uv run mypy src/tbls` clean.
9. `uv build` produces a wheel containing only `src/tbls/` (no `experiments/`);
   `uvx twine check dist/*` passes.
10. **Manual, on the user's machine**: `uv run --group experiments python
    experiments/smoke_run.py` (or `experiments/train.py` against a real
    config) completes without error against
    `experiments/datasets/{biomedical_larger,data_cross_train}.pkl`, and
    `tests/test_real_dataset_smoke.py` passes locally (not just skipped).
    This step is the actual goal of the refactor and must not be skipped
    just because CI is green.

---

## 15. Resolved Decisions (were "Open Questions" in the original draft)

1. **Author / license holder name** for `pyproject.toml` and `LICENSE`.
   `LICENSE` has been added now with placeholder holder
   `TBLS Project Contributors` so the repo has *a* license from its very
   first commit (needed for PyPI regardless). **Action for you**: replace
   with your real name/handle in `LICENSE` and `pyproject.toml`
   `authors = [...]` before the first release if you want personal
   attribution instead of the generic placeholder.
2. **`PairwiseKCCA` / `GraphFuzzyKCCA` as transformers?** — **No, not this
   phase (corrected after checking the actual source).** Both classes are
   two-view estimators (`fit(X1, X2[, y])`, `transform_view1(X_new)`,
   `transform_view2(X_new)`, a no-arg `transform()` returning both training
   projections) — there is no single `X` to hand to a sklearn-standard
   `transform(X)`, so `TransformerMixin` would either be a lie (silently
   drop one view) or invent a non-standard tuple convention that still
   doesn't satisfy `Pipeline`. Add `BaseEstimator` only (for `get_params`/
   `set_params`/`clone()`); leave `Pipeline`/`FeatureUnion` compatibility as
   a genuinely separate, future redesign question (would need e.g. a
   `ColumnTransformer`-style paired-view wrapper), not a one-phase add-on.
3. **`genoptim` / `ensemble` API coupling** — confirmed experimental, not
   part of the stable API, may break across minor versions. `ensemble/*` is
   fully functional after the move (no coupling). `genoptim/*`'s coupling to
   `TBLS` internals is worse than "may change" — the specific attributes it
   needs (`predict(trees=...)`, `mapping_trees`, `tree.selected_features`,
   `tree_params`, `n_map_nodes`, `X_original`) **do not exist at all** on the
   new `TBLS`; the module is migrated (moved, import path fixed, importable)
   but its `TBLS`-dependent functions are not claimed to work, and no test
   asserts they do. Making them work is a follow-up plan that first decides
   whether `TBLS` should grow a subtree-prediction API, not part of this
   refactor. Made explicit at runtime via a `FutureWarning` on import (§5).
4. **Python version floor** — `>=3.10` confirmed (enables `X | Y` union
   syntax already used throughout `othercode/`). Dev machine
   (`.python-version`) is on 3.13; CI matrix (§12.1) tests 3.10 through 3.13
   so the floor is actually verified, not just asserted.
