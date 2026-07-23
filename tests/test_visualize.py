"""Tests for experiments/visualize.py: run discovery and npz side-file loading.

Focused on the path-resolution contract: ``--dir`` may point at any ancestor
of the actual ``.../logs/*.jsonl`` file (the CLI recursively globs
``**/logs/*.jsonl``, e.g. to sweep multiple timestamped runs under one
dataset directory at once) -- npz side-file loading must resolve relative to
each individual jsonl's own ``logs/`` directory, not the top-level ``--dir``
argument.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")  # experiments-only dep; skip otherwise

_EXPERIMENTS_DIR = str(Path(__file__).resolve().parent.parent / "experiments")
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from experiments.visualize import _cohort_predictions  # noqa: E402


def _write_fake_run(base: Path, nested: str, cohort: str = "DM") -> Path:
    """Write a minimal fake run: ``base/<nested>/logs/{run.jsonl,run_..._predictions.npz}``.

    Mimics train.py's actual output layout (``{output_dir}/{model}_{dataset}/
    {timestamp}/logs/...``) so ``nested`` can simulate any depth below the
    directory a user might pass as ``--dir``.
    """
    logs_dir = base / nested / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    npz_name = "run_predictions.npz"
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
    np.savez(
        logs_dir / npz_name,
        **{
            f"{cohort}_fold1_y_true": y_true,
            f"{cohort}_fold1_y_pred": y_pred,
            f"{cohort}_fold1_y_score": y_score,
        },
    )

    event = {
        "record": {
            "extra": {
                "event": "fold_completed",
                "cohort_key": cohort,
                "fold": 1,
                "predictions_file": npz_name,
            }
        }
    }
    (logs_dir / "run.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    return logs_dir


@pytest.mark.parametrize(
    "nested", ["model_dataset/20260101_000000", "a/b/model_dataset/20260101_000000"]
)
def test_cohort_predictions_resolves_npz_regardless_of_dir_depth(
    tmp_path: Path, nested: str
) -> None:
    """Regression guard: npz loading must not assume ``--dir`` == jsonl's grandparent.

    A prior implementation hardcoded ``run_dir / "logs" / predictions_file``,
    which only worked when the ``--dir`` CLI argument was exactly the
    timestamped run directory (one level above ``logs/``). Since
    ``_cohort_predictions``/event discovery recursively globs
    ``**/logs/*.jsonl`` (explicitly supporting ``--dir`` pointing at a
    shallower ancestor, e.g. to sweep every timestamped run under one
    dataset), passing a shallower ``--dir`` silently produced zero loaded
    predictions (swallowed by a bare ``except FileNotFoundError: continue``)
    while still reporting fold events as successfully parsed.
    """
    _write_fake_run(tmp_path, nested)

    preds = _cohort_predictions(tmp_path, "run_tag")

    assert ("run_tag", "DM") in preds
    got = preds[("run_tag", "DM")]
    assert got["y_true"].shape == (4,)
    assert got["y_pred"].shape == (4,)
    assert got["y_score"].shape == (4, 2)
