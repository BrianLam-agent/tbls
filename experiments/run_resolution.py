"""Canonical ``--dir`` → run-directory resolution for the experiment CLIs.

Both :mod:`experiments.visualize` and :mod:`experiments.compare` accept one or
more run directories via ``--dir`` and must turn them into the timestamped run
directory that actually holds ``logs/*.jsonl`` (and the sibling per-cohort
Excel directories used by the comparison Excel). To keep the user-facing rules
identical across CLIs and burden-free, both resolve their ``--dir`` arguments
through :func:`resolve_run_dir`:

- ``--dir <run_name_layer>`` (e.g. ``examples/runs/TBLS Full``): pick the newest
  timestamp subdirectory (lexicographically largest ``YYYYMMDD_HHMMSS`` name).
- ``--dir <run_name_layer>/<timestamp>`` (e.g.
  ``examples/runs/TBLS Full/20260724_074140``): use it directly.

Anything deeper, shallower, or non-timestamp is an error:

- a shallower path (e.g. ``examples/runs``) has no timestamp layer → error;
- a deeper path (e.g. ``.../<timestamp>/logs``) → error;
- a path whose only subdirectory is not a ``YYYYMMDD_HHMMSS`` name → error.

The rules are strict so the user is forced to give a clean input: no shell
globbing, no guessing between multiple timestamps, no silently picking a
stale run. Run names may contain spaces (the underlying directory is
``examples/runs/<run_name>/<timestamp>/`` and Excel sheet names preserve the
space).
"""

from __future__ import annotations

from pathlib import Path
import re

_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def _is_timestamp(name: str) -> bool:
    """Return True if ``name`` matches ``YYYYMMDD_HHMMSS``."""
    return bool(_TIMESTAMP_RE.match(name))


def resolve_run_dir(run_arg: Path) -> Path:
    """Resolve a ``--dir`` argument to the timestamped run directory.

    Args:
        run_arg: A path the user gave on the CLI.

    Returns:
        The run directory ``<run_arg>/<timestamp>`` (the one holding
        ``logs/``).

    Raises:
        NotADirectoryError: ``run_arg`` does not exist or is not a directory.
        FileNotFoundError: ``run_arg`` has no ``YYYYMMDD_HHMMSS`` timestamp
            subdirectory (shallowest valid input is the run-name layer; a
            deeper or shallower path is rejected with this diagnostic).
        ValueError: ``run_arg`` itself is a timestamp dir but the parent is
            not a run-name layer (i.e. the input is too deep, or the parent
            does not look like ``<run_name>``).
    """
    if not run_arg.exists():
        raise NotADirectoryError(f"--dir does not exist: {run_arg}")
    if not run_arg.is_dir():
        raise NotADirectoryError(f"--dir is not a directory: {run_arg}")

    name = run_arg.name
    if _is_timestamp(name):
        # The user gave the timestamp layer directly. Verify the parent looks
        # like a run-name layer (anything but a timestamp is fine; a timestamp
        # parent would mean we are too deep).
        parent = run_arg.parent
        if _is_timestamp(parent.name):
            raise ValueError(
                f"--dir is too deep (a timestamp under another timestamp): {run_arg}. "
                "Give the run-name layer or the run-name/<timestamp> layer."
            )
        if not (run_arg / "logs").is_dir():
            raise FileNotFoundError(
                f"--dir {run_arg} has no `logs/` subdirectory; not a valid run timestamp directory."
            )
        return run_arg

    # The user gave the run-name layer: pick the newest timestamp subdirectory.
    timestamp_dirs = [p for p in run_arg.iterdir() if p.is_dir() and _is_timestamp(p.name)]
    if not timestamp_dirs:
        raise FileNotFoundError(
            f"--dir {run_arg} has no YYYYMMDD_HHMMSS timestamp subdirectory. "
            "Give the run-name layer (the CLI picks the newest) or the "
            "run-name/<timestamp> layer directly."
        )
    newest = max(timestamp_dirs, key=lambda p: p.name)
    if not (newest / "logs").is_dir():
        raise FileNotFoundError(
            f"--dir {newest} (auto-picked newest) has no `logs/` subdirectory; not a valid run."
        )
    return newest


def cohort_excel_dir(run_dir: Path, cohort: str) -> Path:
    """Locate the per-cohort Excel directory for a resolved run directory.

    The training CLI writes per-cohort Excel files at
    ``<run_dir.parent>/<cohort>/<run_dir.name>/`` (i.e. the cohort directory is
    a sibling of the timestamped run directory, and its Excel subdir uses the
    **same timestamp** as ``run_dir``). This helper returns that path and
    validates it exists with the matching timestamp -- the rule the user
    expects: after ``train.py`` finishes, the cohort Excel output is findable
    by timestamp equality.

    Args:
        run_dir: The resolved timestamped run directory (see
            :func:`resolve_run_dir`).
        cohort: The cohort key (e.g. ``"DM"``).

    Returns:
        The cohort's per-key output directory
        ``<run_dir.parent>/<cohort>/<run_dir.name>/``.

    Raises:
        FileNotFoundError: The expected sibling cohort dir + matching-timestamp
            subdir does not exist.
    """
    timestamp = run_dir.name
    cohort_dir = run_dir.parent / cohort / timestamp
    if not cohort_dir.is_dir():
        raise FileNotFoundError(
            f"Per-cohort Excel dir not found: {cohort_dir} (expected "
            f"<run_name>/{cohort}/{timestamp}/, sibling of the run). The run "
            "and its cohort outputs must share a timestamp."
        )
    return cohort_dir
