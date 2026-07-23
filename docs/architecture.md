English | [简体中文](./architecture.zh-CN.md)

# Architecture and repository structure

This document describes how the `tbls` repository is organized, why it is
organized that way, and the design decisions that shape the public API. It is
the source of truth for repository structure; if the tree below drifts from
reality, trust the repository and file an issue/PR to fix this doc.

## 1. Repository layout

```
tbls/                            # repo root
├── pyproject.toml                # package metadata, dependency groups, tool config
├── README.md                     # entry point: install, quickstart, doc index
├── LICENSE                       # Apache-2.0
├── cliff.toml                    # git-cliff changelog configuration
├── .ruff.toml                    # lint/format configuration
├── .pre-commit-config.yaml
├── .github/workflows/            # CI, release, changelog automation
│
├── docs/                         # this documentation set
│
├── src/tbls/                     # the published package (PyPI: `tbls`)
│   ├── __init__.py                # public API surface
│   ├── py.typed                   # PEP 561 marker (package ships type hints)
│   ├── _kernel.py                 # shared: RBF kernel utilities
│   ├── _ifs.py                    # shared: Intuitionistic Fuzzy Set scoring
│   ├── _graph.py                  # shared: graph-Laplacian construction
│   ├── bls.py                     # BroadLearningSystem estimator
│   ├── tbls.py                    # TBLS estimator
│   ├── cca.py                     # PairwiseKCCA + feature-building pipeline
│   ├── gfcca.py                   # GraphFuzzyKCCA + feature-building pipeline
│   ├── genoptim/                  # [experimental] genetic optimizer for tree selection
│   └── ensemble/                  # [experimental] tree-ensemble diversity/selection
│
├── experiments/                  # training CLI + data pipeline (NOT published to PyPI)
│   ├── configs/default.yaml
│   ├── datasets/                  # real .pkl datasets live here (git-ignored)
│   ├── dataprocess.py             # DataLoader: feature selection + resampling
│   ├── evaluate.py                # TBLSEvaluator + TBLSResultSaver
│   ├── classifiers.py             # comparison-algorithm factory
│   ├── train.py                   # typer CLI entry point
│   └── smoke_run.py               # minimal real-dataset sanity check
│
└── tests/                        # pytest suite for the published package
```

## 2. Why `src/` layout

`src/tbls/` (rather than a top-level `tbls/`) prevents accidentally importing
the in-repo package without installing it first — every test and script runs
against the *installed* package, which catches packaging bugs (missing
`__init__.py`, wrong `packages =` glob in `pyproject.toml`, etc.) before they
reach users. This is the layout recommended by the Python Packaging Authority
for publishable packages.

## 3. Why the package/experiments split

The repository implements one research method (TBLS, and the BLS/CCA/GFCCA
building blocks it is built from) but has two very different consumers:

- **Library consumers** (`pip install tbls`) want a small, stable,
  dependency-light package: `numpy` + `scipy` + `scikit-learn` only. They use
  the estimators through the standard `fit`/`predict`/`predict_proba`
  contract, possibly inside `sklearn.pipeline.Pipeline` or
  `GridSearchCV`.
- **This repository's own experiments** need a much heavier, opinionated
  stack: `pandas`, `imbalanced-learn`, `xgboost`, `typer`, `pyyaml`,
  `openpyxl`, a CLI, dataset loaders, Excel report writers, and a large
  comparison-classifier factory (`experiments/classifiers.py`).

Shipping the second stack's dependencies inside the PyPI package would force
every library consumer to install `xgboost`/`pandas`/etc. for code they never
call. So `experiments/` lives outside `src/tbls/`, is never part of the wheel
(`pyproject.toml`'s `[tool.hatch.build.targets.wheel] packages = ["src/tbls"]`
guarantees this — verify with `unzip -l dist/*.whl`), and its dependencies are
an opt-in `uv` dependency group (`uv sync --group experiments`), not part of
`[project.dependencies]`.

## 4. Package internals: shared modules

`TBLS`, `PairwiseKCCA`, and `GraphFuzzyKCCA` all compute an RBF kernel matrix,
and `TBLS`/`GraphFuzzyKCCA` both compute Intuitionistic Fuzzy Set (IFS)
sample-credibility scores. Historically these were three independent
copy-pasted implementations. They are consolidated into three private,
underscore-prefixed modules, imported by the estimators but not part of the
public API:

| Module | Purpose | Used by |
|---|---|---|
| `tbls._kernel` | `rbf_kernel` (general, caller-supplied `gamma`) and `compute_kernel_matrix`/`kernel_distance_matrix` (TBLS's adaptive-gamma variant) | `tbls.py`, `cca.py`, `gfcca.py` |
| `tbls._ifs` | `compute_if_scores_geib` (GEIB formulation, diagonal matrix) and `compute_if_scores_simple` (membership/non-membership/hesitancy, vector) | `tbls.py`, `gfcca.py` |
| `tbls._graph` | `build_graph_laplacian` (kNN intrinsic/penalty) and `build_discriminative_graph_laplacian` (label-only `Lw - beta*Lb`, ported from `GraphFuzzyKCCA`'s inline copy -- intentionally not deduplicated from `gfcca.py`) | `tbls.py` |

These are deliberately **not** re-exported from `tbls/__init__.py` — they are
implementation details that may be refactored (e.g. replaced by a Cython
extension for the `rbf_kernel`/graph-construction hot paths) without a public
API change. If you are extending `tbls`, prefer calling these instead of
re-duplicating kernel/IFS/graph math a fourth time; see
[`development.md`](./development.md).

### A note on numerical fidelity

Because these three modules were extracted from three previously-independent
implementations, they are unit-tested directly (`tests/test_shared_modules.py`)
against a from-scratch reference computation, not just for
shape/finiteness. This matters concretely: an earlier refactor pass introduced
a silent regression in `build_graph_laplacian`'s similarity bandwidth (it used
the median of only the kNN-selected edges instead of the median over *all*
pairwise distances, changing the fitted regularization strength). A
shape/finiteness-only test did not catch it; a bit-for-bit comparison against
an independent reference implementation did. If you touch `_kernel.py`,
`_ifs.py`, or `_graph.py`, keep (or extend) that style of test.

## 5. Estimator contract

Every classifier shipped in `tbls` (`TBLS`, `BroadLearningSystem`) is a full
scikit-learn estimator:

- Inherits `sklearn.base.BaseEstimator` + `ClassifierMixin`.
- Every constructor argument is stored as an identically-named attribute (the
  `sklearn.base.clone()` requirement) — `get_params`/`set_params` are
  inherited from `BaseEstimator`, not hand-written.
- Implements `fit(X, y) -> self`, `predict(X)`, `predict_proba(X)`.
- Sets `self.classes_` / `self.n_classes_` in `fit`.
- Does no file I/O and no logging inside `fit`/`predict` — training/eval
  side effects (Excel reports, progress bars) live in `experiments/`, not the
  package.

`PairwiseKCCA` and `GraphFuzzyKCCA` are feature extractors, not classifiers,
and have a **two-view** API that does not fit sklearn's single-argument
`TransformerMixin.transform(X)` contract:

```python
model.fit(X1, X2)                 # PairwiseKCCA; GraphFuzzyKCCA also takes y
model.transform_view1(X1_new)     # project new view-1 samples
model.transform_view2(X2_new)     # project new view-2 samples
model.transform()                 # no-arg: return both views' *training* projections
```

They inherit `BaseEstimator` only (for `get_params`/`set_params`/`clone()`),
deliberately **not** `TransformerMixin` — a two-argument `fit(X1, X2)` and a
`transform()` that needs to know which of two feature matrices you mean
cannot be squeezed into `transform(X)` without silently dropping a view or
inventing a non-standard calling convention that still would not satisfy
`sklearn.pipeline.Pipeline`. If you need `Pipeline` compatibility for a
two-view model, wrap it yourself (e.g. a small adapter that concatenates or
tuples the two views) rather than expecting it from `tbls` directly. See
[`usage-cca-gfcca.md`](./usage-cca-gfcca.md).

## 6. Experimental subpackages: `tbls.genoptim`, `tbls.ensemble`

`tbls.genoptim` (a genetic optimizer for selecting/weighting `TBLS` trees) and
`tbls.ensemble` (tree-diversity metrics and a generic top-k/threshold
selector) ship inside the package (they need no dependencies beyond
numpy/scipy/scikit-learn, so there is no packaging reason to exclude them),
but are explicitly experimental:

- Each subpackage's `__init__.py` emits a `FutureWarning` on first import.
- `tbls.ensemble` has no coupling to `TBLS` internals and is fully functional.
- `tbls.genoptim.fitness`/`ga_optimizer` reference `TBLS` attributes
  (`mapping_trees`, `tree.selected_features`, a `trees=` kwarg on `predict`)
  that **do not exist** on the current `tbls.tbls.TBLS` — they were carried
  over from an older estimator API and are not verified to run end-to-end.
  See [`experimental-modules.md`](./experimental-modules.md) for the full
  story and what a fix would require.

## 7. Data flow (experiments)

```
experiments/datasets/*.pkl
        │  joblib.load
        ▼
experiments/dataprocess.py::DataLoader   (feature selection + resampling,
        │                                 fit on train split only)
        ▼
tbls.TBLS.fit(X_train, y_train)
        │
        ▼
experiments/evaluate.py::TBLSEvaluator   (sklearn metrics: accuracy, F1,
        │                                 AUROC, balanced accuracy, ...)
        ▼
experiments/evaluate.py::TBLSResultSaver (writes results_dir/.../*.xlsx)
```

`experiments/train.py` wires this into a k-fold cross-validation loop driven
by a YAML config (`experiments/configs/default.yaml`) with typer CLI
overrides. See [`usage-experiments-cli.md`](./usage-experiments-cli.md).

## 8. Release engineering

Versioning, the changelog, and the PyPI publishing pipeline are documented in
[`release-process.md`](./release-process.md). In short: Conventional Commits
on `master` → `git-cliff` (config: `cliff.toml`) turns commit history into a
changelog on tag push → GitHub Actions builds the wheel/sdist, creates a
GitHub Release with the changelog and both artifacts attached, and publishes
to PyPI via Trusted Publishing (OIDC, no stored API token).

## 9. Where to go next

- New to the package? Start with the root [`README.md`](../README.md).
- Want to use an estimator? See `usage-tbls.md`, `usage-bls.md`,
  `usage-cca-gfcca.md`.
- Want to run the training CLI against real data? See
  `usage-experiments-cli.md`.
- Want to contribute code? See [`development.md`](./development.md).
- Want to understand `genoptim`/`ensemble`'s limitations? See
  [`experimental-modules.md`](./experimental-modules.md).
- Want to cut a release? See [`release-process.md`](./release-process.md).
