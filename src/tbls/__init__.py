"""Tree-based Broad Learning System (TBLS) for classification."""

from .bls import BroadLearningSystem
from .cca import (
    PairwiseKCCA,
    build_cca_features,
    project_cca_features,
)
from .gfcca import (
    GraphFuzzyKCCA,
    build_gfcca_features,
)
from .tbls import TBLS

__version__ = "0.1.0"  # kept in sync with pyproject.toml; see design.md §8.2

__all__ = [
    "TBLS",
    "BroadLearningSystem",
    "GraphFuzzyKCCA",
    "PairwiseKCCA",
    "build_cca_features",
    "build_gfcca_features",
    "project_cca_features",
]
