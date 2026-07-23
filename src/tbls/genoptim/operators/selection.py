"""Selection operators for the genetic optimizer.

Standalone (no TBLS coupling). Part of the experimental :mod:`tbls.genoptim`
subpackage.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Chromosome: TypeAlias = NDArray[Any]


class TournamentSelector:
    """Tournament selection operator.

    Args:
        k: Tournament size (number of competing individuals).
    """

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def select(self, population: list[Chromosome], fitness_scores: list[float]) -> list[Chromosome]:
        """Select individuals by holding size-``k`` tournaments.

        Returns:
            A new population (with duplicates) of the same length.
        """
        selected: list[Chromosome] = []
        for _ in range(len(population)):
            competitors = np.random.choice(len(population), self.k, replace=False).tolist()
            fitness_vals = [fitness_scores[i] for i in competitors]
            best_idx = competitors[int(np.argmax(fitness_vals))]
            selected.append(population[best_idx])
        return selected


class RouletteWheelSelector:
    """Roulette-wheel (fitness-proportional) selection operator."""

    def select(
        self,
        population: list[Chromosome],
        fitness_scores: list[float],
    ) -> list[Chromosome]:
        """Select individuals proportional to (shifted) fitness.

        Returns:
            A new population of the same length.
        """
        # Handle negative fitness by shifting.
        adjusted_fitness = (
            np.array(fitness_scores, dtype=np.float64) - np.min(fitness_scores) + 1e-8
        )
        prob = adjusted_fitness / np.sum(adjusted_fitness)
        chosen = np.random.choice(len(population), size=len(population), p=prob).tolist()
        return [population[i] for i in chosen]
