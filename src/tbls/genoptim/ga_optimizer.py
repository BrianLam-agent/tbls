"""Multi-objective genetic optimizer for TBLS tree selection (experimental).

The :meth:`GeneticOptimizer.optimize` method is coupled to TBLS internals that
do not exist on the current :class:`tbls.tbls.TBLS` API (``n_map_nodes``,
``tree_params``, ``X_original``). It is shipped for reuse of the standalone GA
machinery but is **not** verified to run end-to-end. See
``docs/experimental-modules.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tbls.tbls import TBLS

from .encoding import ChromosomeEncoder, PopulationInitializer
from .fitness import MultiObjectiveFitness
from .operators.crossover import SinglePointCrossover, UniformCrossover
from .operators.mutation import BitFlipMutation
from .operators.selection import RouletteWheelSelector, TournamentSelector


class GeneticOptimizer:
    """Multi-objective genetic algorithm optimizer.

    Args:
        n_generations: Number of generations to evolve.
        population_size: Number of individuals per generation.
        crossover_rate: Probability of applying crossover.
        mutation_rate: Probability of applying mutation.
        selection_method: ``'tournament'`` or ``'roulette'``.
        crossover_method: ``'uniform'`` or ``'single_point'``.
        mutation_method: Reserved (currently unused at runtime).
        **kwargs: Extra options, e.g. ``mutation_rate`` override.
    """

    def __init__(
        self,
        n_generations: int = 50,
        population_size: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        selection_method: str = "tournament",
        crossover_method: str = "uniform",
        mutation_method: str = "bit_flip",  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        self.n_generations = n_generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selector = (
            TournamentSelector(k=3) if selection_method == "tournament" else RouletteWheelSelector()
        )
        self.crossover = (
            UniformCrossover() if crossover_method == "uniform" else SinglePointCrossover()
        )
        self.mutator = BitFlipMutation(rate=kwargs.get("mutation_rate", 0.01))

    def optimize(
        self,
        model: TBLS,
        X_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
    ) -> list[int]:
        """Evolve a tree subset and return the best individual's selected trees.

        Note:
            Coupled to TBLS internals (``n_map_nodes``, ``tree_params``,
            ``X_original``, ``y``) that do not exist on the current
            :class:`tbls.tbls.TBLS`. Not verified to run end-to-end; see
            ``docs/experimental-modules.md``.
        """
        # Initialize population.
        encoder = ChromosomeEncoder(model.n_map_nodes)
        initializer = PopulationInitializer(
            self.population_size,
            model.tree_params["bootstrap_ratio"],
        )
        population = initializer.initialize(model.X_original, model.y)
        fitness_calculator = MultiObjectiveFitness(X_val, y_val)

        for _ in range(self.n_generations):
            # Fitness.
            fitness_scores = [
                fitness_calculator.calculate(model, encoder.decode(ind))  # type: ignore[arg-type]
                for ind in population
            ]

            # Selection.
            selected = self.selector.select(population, fitness_scores)

            # Crossover and mutation.
            offspring = []
            for i in range(0, len(selected), 2):
                parent1, parent2 = selected[i], selected[i + 1]
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self.crossover.crossover(parent1, parent2)
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1, parent2])

            # Mutation.
            offspring = [self.mutator.mutate(ind) for ind in offspring]

            # Replacement (elitism).
            combined = population + offspring
            fitness_combined = [
                fitness_calculator.calculate(model, encoder.decode(ind))  # type: ignore[arg-type]
                for ind in combined
            ]
            elite_indices = np.argsort(fitness_combined)[-self.population_size :]
            population = [combined[int(i)] for i in elite_indices]

        # Return the best individual.
        best_idx = int(np.argmax(fitness_scores))
        return encoder.decode(population[best_idx])  # type: ignore[arg-type]

    def _elitism(self, population: list[Any], fitness: list[float], n_elites: int = 2) -> list[Any]:
        """Keep the top ``n_elites`` individuals."""
        elite_indices = np.argsort(fitness)[-n_elites:]
        return [population[int(i)] for i in elite_indices]
