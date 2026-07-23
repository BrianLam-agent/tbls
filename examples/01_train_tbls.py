"""Worked example 01: a single TBLS training run on a real dataset.

This is documentation-by-example: the shortest path from "I have the real
biomedical pkl" to "TBLS fitted and evaluated on a held-out split". It is *not*
a test (the test suite already covers correctness) -- it exists so a new user
or reviewer can see the intended usage pattern in under a minute.

Prerequisites:
    - Install the experiments-only dependencies (heavier than the published
      package)::

          uv sync --group experiments

    - The real dataset must already be present at
      ``experiments/datasets/biomedical_larger.pkl`` (git-ignored; see
      ``experiments/datasets/README.md``).

Expected runtime: a few seconds (1703 samples, 204 features, 10 mapping +
10 enhancement trees).

What it does, step by step:
    1. Load the ``"DM"`` cohort from the pkl, reusing
       :func:`experiments.smoke_run._extract_xy` -- the same defensive
       pkl/dtype/label handling the smoke check uses, rather than reinventing
       it.
    2. A single stratified train/test split (no cross-validation -- keep the
       example minimal; the full CLI in ``experiments/train.py`` does k-fold).
    3. Standardize + Lasso feature selection on the *train* split only, applied
       to the test split via :class:`experiments.dataprocess.DataLoader` (no
       leakage into the test split).
    4. Fit :class:`tbls.TBLS` with Intuitionistic Fuzzy Set sample weighting
       enabled -- TBLS's differentiating feature versus a plain Broad Learning
       System.
    5. Print accuracy, balanced accuracy, and macro F1 on the held-out split
       via :meth:`experiments.evaluate.TBLSEvaluator.calculate_metrics`.

See ``docs/usage-tbls.md`` for the full TBLS API and
``docs/usage-experiments-cli.md`` for the training CLI.
"""

from __future__ import annotations

from pathlib import Path
import sys

# ``experiments/`` is not an installed package: its modules use script-style
# sibling imports (``from dataprocess import ...``), so both the repo root (for
# ``experiments.*`` namespace imports) and the ``experiments/`` directory itself
# (so those sibling imports resolve at import time) must be on ``sys.path``.
# This mirrors what ``tests/conftest.py`` does for the test suite.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.dataprocess import DataLoader  # noqa: E402
from experiments.evaluate import TBLSEvaluator  # noqa: E402
from experiments.smoke_run import _extract_xy  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from tbls import TBLS  # noqa: E402


def main() -> int:
    """Fit one TBLS model on the DM cohort and print held-out metrics.

    Returns:
        0 on success, 1 if the dataset pkl is missing.
    """
    pkl_path = _REPO_ROOT / "experiments" / "datasets" / "biomedical_larger.pkl"
    if not pkl_path.exists():
        print(
            f"Dataset not found at {pkl_path}.\n"
            "Place the real .pkl under experiments/datasets/ (git-ignored); "
            "see experiments/datasets/README.md.",
            file=sys.stderr,
        )
        return 1

    # 1. Load the "DM" cohort (drops label -1, binarizes to {0,1}, coerces
    #    dtype=object -> float64, zeros NaN/Inf -- the standard pkl contract).
    x, y, key = _extract_xy(pkl_path, key="DM")

    # 2. Single stratified split (no CV -- minimal example).
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Standardize + Lasso feature selection, fit on the train split only.
    #    No resampling here (kept minimal); the full CLI exposes SMOTE/etc.
    loader = DataLoader(dataset_name="biomedical_larger", feature_selection="lasso")
    x_tr, y_tr, x_te = loader.preprocess(x_train, y_train, x_test)

    # 4. Fit TBLS with Intuitionistic Fuzzy Set sample weighting enabled -- a
    #    TBLS differentiator versus a plain Broad Learning System. NOTE:
    #    ``use_if_weights=True`` is set explicitly here (it is NOT TBLS's
    #    constructor default, which is False) to showcase the IFS feature.
    #    ``graph_gamma`` is left at its default 0.0 (graph regularization off):
    #    the combination ``use_if_weights=True`` + ``graph_gamma=0.1`` collapses
    #    to all-one-class predictions on this dataset (balanced_accuracy 0.5),
    #    while ``use_if_weights=True`` alone is non-degenerate and matches the
    #    smoke-test reference (~0.92 accuracy). That collapse is flagged as a
    #    separate finding in the plan-03 acceptance report -- not fixed here
    #    (out of scope: this plan adds examples only, no library changes).
    #    ``graph_strategy="discriminative"`` and ``if_strategy="simple"`` are
    #    the constructor defaults.
    model = TBLS(
        n_map_trees=10,
        n_enhance_trees=10,
        use_if_weights=True,
        random_state=42,
    )
    model.fit(x_tr, y_tr)

    # 5. Evaluate on the held-out split. ``calculate_metrics`` provides accuracy
    #    and balanced_accuracy (plus binary F1 as ``f1_score``); macro F1 is
    #    computed separately to match the smoke_run convention.
    y_pred = model.predict(x_te).ravel()
    y_score = model.predict_proba(x_te)
    metrics = TBLSEvaluator.calculate_metrics(y_test, y_pred, y_score)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    print(f"Worked example 01: single TBLS run (cohort={key})")
    print(
        f"  train={x_tr.shape[0]} test={x_te.shape[0]} "
        f"features_in={x.shape[1]} features_selected={x_tr.shape[1]}"
    )
    print(
        f"  accuracy          = {metrics['accuracy']:.4f}\n"
        f"  balanced_accuracy = {metrics['balanced_accuracy']:.4f}\n"
        f"  macro_f1          = {macro_f1:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
