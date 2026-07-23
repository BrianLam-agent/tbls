"""Experimental tree-ensemble selection utilities (may change without notice)."""

import warnings

warnings.warn(
    "tbls.ensemble is experimental: its API may change without notice "
    "between minor versions.",
    category=FutureWarning,
    stacklevel=2,
)

from .diversity_metrics import (  # noqa: E402
    diversity_score,
    feature_entropy_diversity,
    jaccard_similarity,
    pairwise_jaccard_diversity,
)
from .tree_selector import TreeSelector  # noqa: E402

__all__ = [
    "diversity_score",
    "feature_entropy_diversity",
    "jaccard_similarity",
    "pairwise_jaccard_diversity",
    "TreeSelector",
]
