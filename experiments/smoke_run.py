"""Minimal real-dataset smoke test for the ``tbls`` package.

Loads one real sub-dataset from ``experiments/datasets/biomedical_larger.pkl``,
fits a small :class:`tbls.TBLS` on a train/test split, and asserts the
predictions are finite, non-degenerate, and that ``predict_proba`` rows sum to 1.

This is the user's actual acceptance bar ("run the refactored package against
the real dataset"); see ``docs/design.md`` §9.3. The loader is intentionally
defensive: the real pkl stores feature matrices with ``dtype=object`` and a
multi-key dict of sub-datasets.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from tbls import TBLS, BroadLearningSystem


def _extract_xy(
    pkl_path: Path, key: str | None = None
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Load a (sub-)dataset from a pkl and return ``(X, y, key)``.

    Handles both a flat ``{'data':..., 'target':...}`` dict and a multi-key dict
    of sub-datasets (in which case ``key`` selects the sub-dataset, defaulting
    to the first). Feature matrices with ``dtype=object`` are coerced to
    ``float64``; samples with label ``-1`` are dropped; labels are binarized to
    ``{0, 1}`` (matching the legacy ``main.py`` behaviour).
    """
    data = joblib.load(pkl_path)
    sub: dict[str, Any]
    selected_key = key
    if isinstance(data, dict) and "data" in data and "target" in data:
        sub = data
    elif isinstance(data, dict):
        if key is not None:
            sub = data[key]
        else:
            # First sub-dict carrying 'data' and 'target'.
            selected_key, sub = next(
                (k, v)
                for k, v in data.items()
                if isinstance(v, dict) and "data" in v and "target" in v
            )
    else:
        raise TypeError(f"Unsupported pkl top-level type: {type(data)}")

    x = np.asarray(sub["data"], dtype=np.float64)
    y = np.asarray(sub["target"]).ravel()

    # Drop invalid labels, then binarize.
    valid = y != -1
    x, y = x[valid], y[valid]
    y = (y > 0).astype(np.int64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y, selected_key


def run_smoke_check(
    pkl_path: Path | str,
    key: str | None = None,
    max_rows: int | None = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit a small TBLS (and BLS) on a real dataset split and assert sanity.

    Args:
        pkl_path: Path to the real ``.pkl`` dataset.
        key: Optional sub-dataset key for multi-key pkls (default: first).
        max_rows: Cap on rows (subsampled) so the check finishes in seconds on
            large files. ``None`` uses all rows.
        random_state: Seed for the split and the estimators.

    Returns:
        A dict with ``accuracy``, ``macro_f1``, ``bls_accuracy``, ``n_train``,
        ``n_test``, ``n_features``, and ``key``.

    Raises:
        AssertionError: If predictions contain NaN/Inf, ``predict_proba`` rows
            do not sum to 1, or predictions are degenerate (single class).
    """
    pkl_path = Path(pkl_path)
    x, y, key = _extract_xy(pkl_path, key=key)

    # Subsample for speed if the dataset is large.
    if max_rows is not None and x.shape[0] > max_rows:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(x.shape[0], size=max_rows, replace=False)
        x, y = x[idx], y[idx]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Small TBLS so it finishes in seconds.
    model = TBLS(n_map_trees=10, n_enhance_trees=10, random_state=random_state)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)
    pred = model.predict(x_test)

    # Sanity assertions.
    assert np.isfinite(proba).all(), "predict_proba produced NaN/Inf"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3), "predict_proba rows do not sum to 1"
    assert len(np.unique(pred)) > 1, "predictions are degenerate (single class)"

    accuracy = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))

    # Also smoke-test BroadLearningSystem on the same split.
    bls = BroadLearningSystem(
        n_feature_groups=5,
        n_feature_nodes_per_group=20,
        n_enhancement_groups=5,
        n_enhancement_nodes_per_group=20,
        random_state=random_state,
    )
    bls.fit(x_train, y_train)
    bls_pred = bls.predict(x_test)
    bls_accuracy = float(accuracy_score(y_test, bls_pred))

    return {
        "key": key,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "bls_accuracy": bls_accuracy,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features": int(x.shape[1]),
    }


def main() -> int:
    """Run the smoke check against ``experiments/datasets/biomedical_larger.pkl``."""
    pkl_path = Path(__file__).parent / "datasets" / "biomedical_larger.pkl"
    if not pkl_path.exists():
        print(f"Dataset not found at {pkl_path}", file=sys.stderr)
        return 1
    result = run_smoke_check(pkl_path)
    print(
        "TBLS smoke check OK | "
        f"key={result['key']} "
        f"acc={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
        f"bls_acc={result['bls_accuracy']:.4f} "
        f"train={result['n_train']} test={result['n_test']} "
        f"features={result['n_features']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
