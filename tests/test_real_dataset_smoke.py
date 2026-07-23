"""Real-dataset smoke test.

Skipped (not failed) when ``experiments/datasets/biomedical_larger.pkl`` is
absent - i.e. in CI. On the dev machine, where the data exists, it runs the
shared :func:`experiments.smoke_run.run_smoke_check`.
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


def test_tbls_fits_real_dataset() -> None:
    from experiments.smoke_run import run_smoke_check

    result = run_smoke_check(REAL_DATA)
    assert result["accuracy"] > 0.0
    assert result["macro_f1"] > 0.0
