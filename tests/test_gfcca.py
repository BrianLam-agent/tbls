"""GraphFuzzyKCCA tests (two-view API)."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone

from tbls import GraphFuzzyKCCA


def test_gfcca_fit_transform() -> None:
    rng = np.random.RandomState(0)
    n = 50
    x1 = rng.randn(n, 10)
    x2 = rng.randn(n, 8)
    y = rng.randint(0, 2, size=n).astype(np.int64)
    model = GraphFuzzyKCCA(k=3, reg_lambda=0.1, kernel_gamma=0.1)
    model.fit(x1, x2, y)
    z1, z2 = model.transform()
    assert z1.shape == (n, 3)
    assert z2.shape == (n, 3)

    p1 = model.transform_view1(x1[:5])
    assert p1.shape == (5, 3)
    assert np.isfinite(p1).all()


def test_gfcca_get_set_params() -> None:
    model = GraphFuzzyKCCA(k=4, graph_gamma=0.2)
    params = model.get_params()
    assert params["k"] == 4
    assert clone(model).get_params() == params
    model.set_params(k=6)
    assert model.k == 6
