English | [简体中文](./README.zh-CN.md)

# tbls

[![CI](https://github.com/BrianLam-agent/tbls/actions/workflows/ci.yml/badge.svg)](https://github.com/BrianLam-agent/tbls/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tbls.svg)](https://pypi.org/project/tbls/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/tbls.svg)](https://pypi.org/project/tbls/)

Tree-based Broad Learning System (TBLS) for classification, with a fully
scikit-learn-compatible API.

`tbls` packages a Tree-based Broad Learning System classifier (`TBLS`) built
from small regression trees arranged in the Broad Learning System's
two-stage "mapping → enhancement" architecture, trained with a closed-form
ridge solve, plus the building blocks it is composed from: a classic
random-weight `BroadLearningSystem`, and two-view kernel CCA feature
extractors (`PairwiseKCCA`, `GraphFuzzyKCCA`) with optional Intuitionistic
Fuzzy Set sample weighting and graph-Laplacian regularization.

## Installation

```bash
pip install tbls
```

The published package depends only on `numpy`, `scipy`, and `scikit-learn`.

## Quickstart

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from tbls import TBLS

X, y = make_classification(n_samples=200, n_features=20, random_state=0)

model = TBLS(n_map_trees=10, n_enhance_trees=10, random_state=0)
model.fit(X, y)
print(model.predict(X[:5]))
print(model.predict_proba(X[:5]))

print(cross_val_score(model, X, y, cv=3))
```

For runnable worked examples against the real biomedical dataset (a single
TBLS run, and a TBLS `--grid` hyperparameter sweep), see
[`examples/`](examples/README.md). They call the same experiment internals as
the training CLI and finish in seconds.

## What's in the box

| Component | What it is |
|---|---|
| `tbls.TBLS` | Tree-based Broad Learning System classifier, with optional IFS sample weighting and graph-Laplacian regularization. |
| `tbls.BroadLearningSystem` | Classic random-weight Broad Learning System with true incremental-enhancement learning. |
| `tbls.PairwiseKCCA` | Two-view regularized kernel CCA feature extractor. |
| `tbls.GraphFuzzyKCCA` | Two-view kernel CCA with IFS sample credibility and discriminative graph-embedding regularization. |
| `tbls.genoptim` *(experimental)* | Genetic-algorithm tree selection — encoding/operators are functional, `TBLS`-coupled fitness/optimizer are not (see docs). |
| `tbls.ensemble` *(experimental)* | Tree-diversity metrics and a generic fitness-based tree/subset selector — fully functional. |

All classifiers (`TBLS`, `BroadLearningSystem`) are standard
`sklearn.base.BaseEstimator`/`ClassifierMixin` implementations and work with
`cross_val_score`, `GridSearchCV`, `Pipeline`, etc.

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Repository structure, why the package/experiments split exists, shared-module design, the estimator contract. |
| [`docs/usage-tbls.md`](docs/usage-tbls.md) | `TBLS` tutorial: parameters, IFS/graph regularization, incremental layers, reproducibility, performance notes. |
| [`docs/usage-bls.md`](docs/usage-bls.md) | `BroadLearningSystem` tutorial: parameters, class imbalance, Woodbury incremental enhancement. |
| [`docs/usage-cca-gfcca.md`](docs/usage-cca-gfcca.md) | `PairwiseKCCA`/`GraphFuzzyKCCA` tutorial: two-view API, multi-view pipelines, why there's no `Pipeline` support. |
| [`docs/usage-experiments-cli.md`](docs/usage-experiments-cli.md) | Full reference for the `experiments/` pipeline: training/visualize/compare CLIs, YAML config (incl. `run_name`, `grid:`), `--dir` resolution rule, every output file, the JSONL event schema, and `feature_selection`/`resampling` enumerations. |
| [`docs/usage-multiview-fusion.md`](docs/usage-multiview-fusion.md) | Multi-view pkl data contract and CCA/GFCCA fusion-group config (convention only -- not yet implemented; see `docs/plan/02-*`). |
| [`docs/usage-figures-and-calibration.md`](docs/usage-figures-and-calibration.md) | Why `TBLS`'s uncalibrated `predict_proba` produces sharp PR-curve cliffs (the `0.5` score-density plateau), the math, and a self-contained reproducer — read when a PR plot looks pathological. |
| [`docs/experimental-modules.md`](docs/experimental-modules.md) | What works and what doesn't in `tbls.genoptim`/`tbls.ensemble`, and why. |
| [`docs/development.md`](docs/development.md) | Local dev setup, conventions, how to add a new estimator, docs/translation structure. |
| [`docs/release-process.md`](docs/release-process.md) | Semantic versioning, changelog generation, the tag-triggered CI/CD release pipeline, PyPI publishing. |

## Development

```bash
uv sync --group dev --group experiments   # install everything
ruff check .                              # lint
ruff format --check .                     # format check
mypy src/tbls                             # type check
pytest                                     # tests
```

See [`docs/development.md`](docs/development.md) for the full guide.

## Contributing

Issues and pull requests are welcome. Please read
[`docs/development.md`](docs/development.md) first for project conventions
(Conventional Commits, the estimator contract, docs/translation structure).

## References

The GEIB intuitionistic-fuzzy-set formulation used by `TBLS` follows
Chen et al., *IEEE Transactions on Fuzzy Systems*, 2025. The Broad Learning
System architecture follows the original BLS literature.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
