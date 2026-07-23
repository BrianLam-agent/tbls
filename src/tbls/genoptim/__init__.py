"""Experimental genetic optimizer for TBLS tree selection.

Warning:
    The TBLS-coupled functions in ``fitness.py``/``ga_optimizer.py`` reference
    attributes (``mapping_trees``, ``tree.selected_features``, a ``trees=``
    kwarg on ``predict``) that do not exist on :class:`tbls.tbls.TBLS`. They are
    shipped for reference/reuse of the standalone GA machinery (``encoding.py``,
    ``operators/``) but are not verified to run end-to-end. See
    ``docs/design.md`` §15.3.
"""

import warnings

warnings.warn(
    "tbls.genoptim is experimental and its TBLS-coupled functions are "
    "not verified against the current tbls.tbls.TBLS API; see "
    "docs/design.md section 15.3.",
    category=FutureWarning,
    stacklevel=2,
)

from .encoding import ChromosomeEncoder, PopulationInitializer  # noqa: E402
from .fitness import MultiObjectiveFitness  # noqa: E402
from .ga_optimizer import GeneticOptimizer  # noqa: E402

__all__ = [
    "ChromosomeEncoder",
    "GeneticOptimizer",
    "MultiObjectiveFitness",
    "PopulationInitializer",
]
