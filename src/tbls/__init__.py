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
"""Tree-based Broad Learning System (TBLS) for classification."""

from .bls import BroadLearningSystem
from .cca import (
    PairwiseKCCA,
    build_cca_features,
    project_cca_features,
)
from .gfcca import (
    GraphFuzzyKCCA,
    build_gfcca_features,
)
from .tbls import TBLS, build_tbls_variant

__version__ = "0.1.0"  # kept in sync with pyproject.toml; see docs/release-process.md
__copyright__ = "Copyright 2026 BrianLam"
__license__ = "Apache-2.0"

__all__ = [
    "TBLS",
    "BroadLearningSystem",
    "GraphFuzzyKCCA",
    "PairwiseKCCA",
    "build_cca_features",
    "build_gfcca_features",
    "build_tbls_variant",
    "project_cca_features",
]
