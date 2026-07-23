# Copyright 2026 BrianLam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental genetic optimizer for TBLS tree selection.

Warning:
    The TBLS-coupled functions in ``fitness.py``/``ga_optimizer.py`` reference
    attributes (``mapping_trees``, ``tree.selected_features``, a ``trees=``
    kwarg on ``predict``) that do not exist on :class:`tbls.tbls.TBLS`. They are
    shipped for reference/reuse of the standalone GA machinery (``encoding.py``,
    ``operators/``) but are not verified to run end-to-end. See
    ``docs/experimental-modules.md``.
"""

import warnings

warnings.warn(
    "tbls.genoptim is experimental and its TBLS-coupled functions are "
    "not verified against the current tbls.tbls.TBLS API; see "
    "docs/experimental-modules.md.",
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
