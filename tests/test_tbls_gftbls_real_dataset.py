"""Real-data regression: GFTBLS no longer numerically collapses on the DM cohort.

Verifies the primary acceptance gate for Plan 07
(``docs/plan/07-fix-ifs-simple-membership-bandwidth-collapse.md``): the
combination ``TBLS(use_if_weights=True, graph_gamma=0.1)`` (GFTBLS) must no
longer degrade to all-one-class predictions on the ``"DM"`` cohort of
``experiments/datasets/biomedical_larger.pkl`` after the
``compute_if_scores_simple`` Gaussian-membership bandwidth relativization fix.

Before the fix, this exact configuration collapsed to ``balanced_accuracy
= 0.5`` (single-class output) because the absolute ``sigma_if`` membership
bandwidth underflowed ``mu`` to numerical zero for every sample, driving the
IFS weight vector ``s`` to the ``min_weight`` clip uniformly and letting the
graph term dominate a ridge solve decoupled from ``y``.

Skipped (not failed) when the pkl is absent -- i.e. in CI -- exactly
mirroring ``tests/test_real_dataset_smoke.py`` (the dev machine has the data).
"""

from __future__ import annotations

from pathlib import Path

import pytest

DATASET_DIR = Path(__file__).parent.parent / "experiments" / "datasets"
REAL_DATA = DATASET_DIR / "biomedical_larger.pkl"

pytestmark = pytest.mark.skipif(
    not REAL_DATA.exists(),
    reason="real dataset not present (expected in CI, present on dev machine)",
)


def test_gftbls_does_not_collapse_on_dm_cohort() -> None:
    from experiments.smoke_run import _extract_xy
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from tbls import TBLS

    x, y, key = _extract_xy(REAL_DATA, key="DM")
    assert key == "DM"
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    scaler = StandardScaler().fit(x_tr)
    x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)

    model = TBLS(
        n_map_trees=10,
        n_enhance_trees=10,
        use_if_weights=True,
        graph_gamma=0.1,
        random_state=0,
    )
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te).ravel()

    # Hard non-degeneracy gate: the model must predict more than one class.
    assert len(np.unique(pred)) > 1, (
        f"degenerate single-class predictions: {np.unique(pred).tolist()}"
    )
    # Acceptance gate: balanced accuracy clearly above the degenerate 0.5 floor.
    bal_acc = balanced_accuracy_score(y_te, pred)
    assert bal_acc > 0.6, f"balanced_accuracy={bal_acc:.4f} (threshold 0.6)"
