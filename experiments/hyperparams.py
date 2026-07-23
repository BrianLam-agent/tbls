"""Centralized, directly-editable default hyperparameters and grid-search axes.

Edit the ``*_DEFAULTS`` dicts to change the single-run default configuration;
edit the ``*_GRID`` dicts to change what ``--grid`` sweeps. Both are plain
Python dicts, not a YAML/CLI surface, by design -- this mirrors how the values
were originally tuned (as module-level constants), matching the request to keep
this directly editable rather than routed through another layer of config.

The grid values below are a **starting example** (small, roughly
order-of-magnitude neighbors of the user's tuned point estimates), not a claim
that this is the "correct" search space -- tune them here directly.
"""

from __future__ import annotations

BLS_DEFAULTS: dict = {
    "n_feature_groups": 30,
    "n_feature_nodes_per_group": 40,
    "n_enhancement_groups": 1,
    "n_enhancement_nodes_per_group": 500,
    "map_func": "relu",
    "enhance_func": "relu",
    "reg_param": 1.0,
}
BLS_GRID: dict = {
    "n_feature_groups": [15, 30, 60],
    "n_feature_nodes_per_group": [20, 40, 80],
    "reg_param": [0.1, 1.0, 10.0],
}

TBLS_DEFAULTS: dict = {
    "n_map_trees": 10,
    "n_enhance_trees": 10,
    "tree_max_depth": 5,
    "tree_min_samples_split": 3,
    "tree_max_features_ratio": 0.7,
    "reg_param": 1e-8,
    # graph_strategy/if_strategy intentionally left at TBLS's own defaults
    # ("discriminative"/"simple") rather than repeated here.
}
TBLS_GRID: dict = {
    "n_map_trees": [10, 20, 40],
    "n_enhance_trees": [10, 20, 40],
    "reg_param": [1e-8, 1e-4, 1e-2],
}

# CCA/GFCCA fusion hyperparameters (multi-view pipelines; see
# docs/usage-multiview-fusion.md). Keyword names match
# tbls.cca.build_cca_features / tbls.gfcca.build_gfcca_features exactly (these
# dicts are passed as **kwargs in experiments/multiview.py::fuse_views).
CCA_DEFAULTS: dict = {
    "cca_k": 15,
    "cca_lambda": 0.1,
    "kernel_gamma": 1.0,
}
CCA_GRID: dict = {
    "cca_k": [7, 15, 25],
    "cca_lambda": [0.01, 0.1, 1.0],
}

GFCCA_DEFAULTS: dict = {
    "cca_k": 15,
    "cca_lambda": 0.1,
    "kernel_gamma": 1.0,
    "graph_gamma": 0.5,
    "discriminative_beta": 0.3,
    "sigma_if": 1.0,
    "delta_if": 0.5,
    # sigma_graph is a documented-dead GraphFuzzyKCCA parameter (reserved,
    # unused) -- intentionally not included here.
}
GFCCA_GRID: dict = {
    "graph_gamma": [0.1, 0.5, 1.0],
    "discriminative_beta": [0.1, 0.3, 0.5],
}
