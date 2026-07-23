"""Crossover operators for the genetic optimizer.

Standalone (no TBLS coupling). Part of the experimental :mod:`tbls.genoptim`
subpackage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Chromosome = NDArray[Any]


class SinglePointCrossover:
    """Single-point crossover operator."""

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        """Single-point crossover: swap gene segments at a random cut point.

        Returns:
            A pair ``(child1, child2)``.
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have same length")
        crossover_point = int(np.random.randint(1, len(parent1) - 1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2


class UniformCrossover:
    """Uniform crossover operator.

    Args:
        swap_prob: Per-gene probability of taking the gene from ``parent2``.
    """

    def __init__(self, swap_prob: float = 0.5) -> None:
        self.swap_prob = swap_prob

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        """Uniform crossover: each gene swapped independently with ``swap_prob``.

        Returns:
            A pair ``(child1, child2)``.
        """
        mask = np.random.rand(len(parent1)) < self.swap_prob
        child1 = np.where(mask, parent2, parent1)
        child2 = np.where(mask, parent1, parent2)
        return child1, child2
