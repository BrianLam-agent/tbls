"""tbls.ensemble tests (standalone tree-selector + diversity metrics)."""

from __future__ import annotations

import numpy as np

from tbls.ensemble import TreeSelector, diversity_score


def test_tree_selector_top_k() -> None:
    fitness = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.8}
    sel = TreeSelector(selection_method="top_k", weight_method="performance")
    sel.fit(fitness)
    selected = sel.get_selected_trees()
    weights = sel.get_weights()
    assert len(selected) == 2  # top 50% of 4
    assert len(weights) == 2
    assert np.isclose(weights.sum(), 1.0)


def test_tree_selector_threshold_uniform() -> None:
    fitness = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.8}
    sel = TreeSelector(selection_method="threshold", weight_method="uniform")
    sel.fit(fitness)
    assert len(sel.get_selected_trees()) >= 1


def test_tree_selector_diversity_weights() -> None:
    fitness = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.8}
    diversity = {0: 0.2, 1: 0.7, 2: 0.5, 3: 0.6}
    sel = TreeSelector(selection_method="top_k", weight_method="diversity")
    sel.fit(fitness, diversity_scores=diversity)
    weights = sel.get_weights()
    assert np.isclose(weights.sum(), 1.0)


def test_diversity_score_jaccard() -> None:
    sets = [{0, 1, 2}, {1, 2, 3}, {3, 4}]
    score = diversity_score(sets, method="jaccard")
    assert 0.0 <= score <= 1.0


def test_diversity_score_entropy() -> None:
    sets = [{0, 1, 2}, {1, 2, 3}, {3, 4}]
    score = diversity_score(sets, method="entropy")
    assert score >= 0.0
