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
"""Experimental tree-ensemble selection utilities (may change without notice)."""

import warnings

warnings.warn(
    "tbls.ensemble is experimental: its API may change without notice between minor versions.",
    category=FutureWarning,
    stacklevel=2,
)

from .diversity_metrics import (  # noqa: E402
    diversity_score,
    feature_entropy_diversity,
    jaccard_similarity,
    pairwise_jaccard_diversity,
)
from .tree_selector import TreeSelector  # noqa: E402

__all__ = [
    "TreeSelector",
    "diversity_score",
    "feature_entropy_diversity",
    "jaccard_similarity",
    "pairwise_jaccard_diversity",
]
