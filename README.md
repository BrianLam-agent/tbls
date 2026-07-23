# tbls

Tree-based Broad Learning System (TBLS) for classification, with a sklearn-compatible API.

This package refactors the research implementation into an installable Python
package. The core estimators (`TBLS`, `BroadLearningSystem`) and feature
extractors (`PairwiseKCCA`, `GraphFuzzyKCCA`) are sklearn-compatible
estimators usable with `cross_val_score`, `GridSearchCV`, etc.

## Installation

The package is not yet on PyPI. For local development:

```bash
uv pip install -e .
```

or, with the full experiment tooling:

```bash
uv sync --group dev --group experiments
```

The published package depends only on `numpy`, `scipy` and `scikit-learn`.

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

## Features

- `tbls.TBLS` — tree-based BLS with optional Intuitionistic Fuzzy Set (IFS)
  sample weighting and graph-Laplacian regularization.
- `tbls.BroadLearningSystem` — classic random-weight Broad Learning System.
- `tbls.PairwiseKCCA` / `tbls.GraphFuzzyKCCA` — two-view kernel CCA feature
  extractors (note: two-view API, not sklearn `Pipeline`-compatible; see
  `docs/design.md` §15.2).

### Experimental subpackages

`tbls.genoptim` and `tbls.ensemble` are **experimental**. Their public API may
change without notice between minor versions. In particular, `tbls.genoptim`'s
`fitness.py` / `ga_optimizer.py` are coupled to TBLS internals that do not
exist on the current `TBLS` API and are therefore not verified to run
end-to-end; see `docs/design.md` §15.3.

## Development

```bash
uv sync --group dev --group experiments   # install everything
ruff check .                              # lint
ruff format --check .                     # format check
mypy src/tbls                             # type check
pytest                                    # tests
```

## References

The GEIB intuitionistic-fuzzy-set formulation used by `TBLS` follows
Chen et al., *IEEE Transactions on Fuzzy Systems*, 2025. The Broad Learning
System architecture follows the original BLS literature.

## License

MIT — see [LICENSE](LICENSE).
