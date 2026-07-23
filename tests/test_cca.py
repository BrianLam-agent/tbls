"""PairwiseKCCA tests (two-view API; no TransformerMixin/Pipeline per design 15.2)."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone

from tbls import PairwiseKCCA


def test_pairwise_kcca_fit_transform() -> None:
    rng = np.random.RandomState(0)
    n = 60
    x1 = rng.randn(n, 10)
    x2 = rng.randn(n, 8)
    cca = PairwiseKCCA(k=3, reg_lambda=0.1, kernel_gamma=0.1)
    cca.fit(x1, x2)
    z1, z2 = cca.transform()
    assert z1.shape == (n, 3)
    assert z2.shape == (n, 3)

    # Held-out projection.
    x1_new = rng.randn(10, 10)
    x2_new = rng.randn(10, 8)
    p1 = cca.transform_view1(x1_new)
    p2 = cca.transform_view2(x2_new)
    assert p1.shape == (10, 3)
    assert p2.shape == (10, 3)
    assert np.isfinite(p1).all() and np.isfinite(p2).all()


def test_pairwise_kcca_get_set_params() -> None:
    cca = PairwiseKCCA(k=5, reg_lambda=0.2, kernel_gamma=0.3)
    params = cca.get_params()
    assert params["k"] == 5
    assert clone(cca).get_params() == params
    cca.set_params(k=7)
    assert cca.k == 7
