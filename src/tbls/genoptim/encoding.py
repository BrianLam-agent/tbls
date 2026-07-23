"""Chromosome encoding and population initialization for the genetic optimizer.

Standalone (no TBLS coupling). Part of the experimental :mod:`tbls.genoptim`
subpackage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ChromosomeEncoder:
    """Encode tree-subset selections as binary chromosomes.

    Args:
        max_trees: Maximum number of trees (chromosome length).
    """

    def __init__(self, max_trees: int) -> None:
        self.max_trees = max_trees

    def encode(self, selected_indices: list[int]) -> NDArray[np.bool_]:
        """Encode selected tree indices into a binary chromosome."""
        chromosome = np.zeros(self.max_trees, dtype=np.bool_)
        chromosome[selected_indices] = True
        return chromosome

    def decode(self, chromosome: NDArray[np.bool_]) -> list[int]:
        """Decode a binary chromosome back into selected tree indices."""
        return np.where(chromosome)[0].tolist()


class PopulationInitializer:
    """Initialize a population via bootstrap sampling.

    Args:
        n_population: Number of individuals in the population.
        bootstrap_ratio: Fraction of samples drawn per individual.
    """

    def __init__(self, n_population: int, bootstrap_ratio: float) -> None:
        self.n_population = n_population
        self.bootstrap_ratio = bootstrap_ratio

    def initialize(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],  # noqa: ARG002
    ) -> list[NDArray[np.intp]]:
        """Generate the initial population of bootstrap index arrays.

        Args:
            X: Samples of shape ``(n_samples, n_features)`` (only ``n_samples``
                is used to size the bootstrap draw).
            y: Labels, accepted for API symmetry and currently unused.

        Returns:
            A list of ``n_population`` bootstrap index arrays.
        """
        n_samples = X.shape[0]
        population: list[NDArray[np.intp]] = []
        for _ in range(self.n_population):
            indices = np.random.choice(
                n_samples, int(n_samples * self.bootstrap_ratio), replace=True
            )
            population.append(indices)
        return population
