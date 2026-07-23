"""Mutation operators for the genetic optimizer.

Standalone (no TBLS coupling). Part of the experimental :mod:`tbls.genoptim`
subpackage.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Chromosome: TypeAlias = NDArray[Any]


class BitFlipMutation:
    """Bit-flip mutation operator for binary encodings.

    Args:
        rate: Per-gene probability of flipping.
    """

    def __init__(self, rate: float = 0.01) -> None:
        self.rate = rate

    def mutate(self, individual: Chromosome) -> Chromosome:
        """Flip each gene independently with probability ``rate``.

        Returns:
            The mutated individual.
        """
        mutation_mask = np.random.rand(len(individual)) < self.rate
        return np.logical_xor(individual, mutation_mask)


class GaussianMutation:
    """Gaussian mutation operator for real-valued encodings.

    Args:
        rate: Per-gene probability of perturbing.
        scale: Standard deviation of the Gaussian noise.
    """

    def __init__(self, rate: float = 0.1, scale: float = 0.1) -> None:
        self.rate = rate
        self.scale = scale

    def mutate(self, individual: Chromosome) -> Chromosome:
        """Add Gaussian noise to each gene with probability ``rate``.

        Returns:
            The mutated individual.
        """
        mutation_mask = np.random.rand(len(individual)) < self.rate
        noise = np.random.normal(0, self.scale, len(individual))
        return individual + mutation_mask * noise


class AdaptiveMutation:
    """Mutation operator that adjusts its rate from population diversity.

    Args:
        min_rate: Lower bound for the mutation rate.
        max_rate: Upper bound for the mutation rate.
    """

    def __init__(self, min_rate: float = 0.01, max_rate: float = 0.3) -> None:
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.rate: float = min_rate

    def update_rate(self, population_diversity: float) -> None:
        """Raise the mutation rate when population diversity is low."""
        self.rate = self.max_rate - (self.max_rate - self.min_rate) * population_diversity
