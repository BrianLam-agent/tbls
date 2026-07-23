"""Diversity metrics for tree feature subsets.

Standalone utilities (no TBLS coupling) used by the experimental tree-selection
tooling in :mod:`tbls.ensemble`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy  # type: ignore[import-untyped]


def jaccard_similarity(set_a: set[int], set_b: set[int]) -> float:
    """Jaccard similarity between two feature subsets."""
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union != 0 else 0.0


def pairwise_jaccard_diversity(feature_sets: list[set[int]]) -> float:
    """Average pairwise Jaccard diversity (``1 - similarity``) across subsets."""
    n = len(feature_sets)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1 - jaccard_similarity(feature_sets[i], feature_sets[j])
    return total / (n * (n - 1) / 2)


def feature_entropy_diversity(feature_sets: list[set[int]], all_features: int) -> float:
    """Feature-occurrence-frequency entropy across subsets."""
    feature_counts = np.zeros(all_features, dtype=np.float64)
    for features in feature_sets:
        for f in features:
            feature_counts[f] += 1
    prob = feature_counts / len(feature_sets)
    prob = prob[prob > 0]  # drop zero-probability entries
    return float(entropy(prob, base=2))


def diversity_score(feature_sets: list[set[int]], method: str = "jaccard") -> float:
    """Unified diversity-score interface.

    Args:
        feature_sets: List of feature-index subsets.
        method: ``'jaccard'`` for pairwise Jaccard diversity, ``'entropy'`` for
            feature-occurrence entropy.

    Returns:
        The diversity score.
    """
    sets = [set(f) for f in feature_sets]
    if method == "jaccard":
        return pairwise_jaccard_diversity(sets)
    if method == "entropy":
        all_features = max(max(f) for f in feature_sets) + 1
        return feature_entropy_diversity(sets, all_features)
    raise ValueError(f"Unsupported diversity method: {method}")
