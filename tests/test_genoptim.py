"""tbls.genoptim experimental-subpackage tests.

Only the TBLS-independent pieces (encoding, operators) are unit-tested. The
TBLS-coupled ``MultiObjectiveFitness.calculate`` / ``GeneticOptimizer.optimize``
are intentionally NOT tested end-to-end (see docs/experimental-modules.md).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tbls.genoptim.encoding import ChromosomeEncoder, PopulationInitializer
from tbls.genoptim.operators.crossover import SinglePointCrossover, UniformCrossover
from tbls.genoptim.operators.mutation import BitFlipMutation
from tbls.genoptim.operators.selection import RouletteWheelSelector, TournamentSelector


def test_genoptim_import_warns() -> None:
    """Importing tbls.genoptim emits a FutureWarning."""
    # Drop cached modules so the package __init__ re-executes.
    for mod in [m for m in sys.modules if m == "tbls.genoptim" or m.startswith("tbls.genoptim.")]:
        del sys.modules[mod]
    with pytest.warns(FutureWarning, match="experimental"):
        import tbls.genoptim  # noqa: F401


def test_chromosome_encoder_roundtrip() -> None:
    enc = ChromosomeEncoder(max_trees=10)
    chrom = enc.encode([1, 3, 7])
    assert chrom.dtype == np.bool_
    assert chrom.sum() == 3
    assert enc.decode(chrom) == [1, 3, 7]


def test_population_initializer_shape() -> None:
    np.random.seed(0)
    x = np.random.randn(100, 5)
    y = np.zeros(100, dtype=np.int64)
    pop = PopulationInitializer(n_population=4, bootstrap_ratio=0.5).initialize(x, y)
    assert len(pop) == 4
    for ind in pop:
        assert ind.shape[0] == 50


def test_operators_on_synthetic() -> None:
    rng = np.random.RandomState(0)
    pop = [rng.randint(0, 2, size=8).astype(np.bool_) for _ in range(6)]
    fitness = [float(rng.rand()) for _ in range(6)]

    selected = TournamentSelector(k=3).select(pop, fitness)
    assert len(selected) == 6
    selected_r = RouletteWheelSelector().select(pop, fitness)
    assert len(selected_r) == 6

    c1, _ = UniformCrossover().crossover(pop[0], pop[1])
    assert c1.shape == pop[0].shape
    c3, _ = SinglePointCrossover().crossover(pop[0], pop[1])
    assert c3.shape == pop[0].shape

    mutated = BitFlipMutation(rate=0.5).mutate(pop[0])
    assert mutated.shape == pop[0].shape
