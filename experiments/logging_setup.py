"""Dual-sink loguru logging setup for the experiments CLI.

``configure_logging`` replaces loguru's default handler with two sinks:

- a human-readable, colorized **stdout** sink at level ``INFO`` (so existing
  eyeballed output stays readable);
- a structured **JSONL file** sink at level ``DEBUG`` with loguru's
  ``serialize=True`` (one JSON object per record, including the bound event
  dict under ``record["extra"]``) at
  ``{output_dir}/logs/{dataset}_{timestamp}.jsonl``.

Stdlib ``logging`` is intercepted (:class:`InterceptHandler`) so warnings
emitted via ``logging.getLogger`` (e.g. from ``experiments/evaluate.py``) flow
through the same two sinks. Call this once near the top of ``train.py``'s CLI
entrypoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """Forward stdlib :mod:`logging` records to loguru.

    Lets modules that still use ``logging.getLogger(__name__)`` (e.g.
    ``experiments/evaluate.py``'s probability-metric warnings) appear on both
    the stdout and JSONL sinks without each module needing to import loguru.

    Adapted from the loguru stdlib-interception recipe.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a stdlib record through the loguru logger.

        Args:
            record: The stdlib log record to forward.
        """
        try:
            level: str | int = logger.level(record.levelname).name
        except (ValueError, AttributeError):
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(output_dir: Path, dataset: str, timestamp: str) -> None:
    """Configure loguru's two sinks and intercept stdlib logging.

    Removes loguru's default handler, then adds the stdout (INFO, colorized)
    and JSONL-file (DEBUG, ``serialize=True``) sinks. The JSONL file path is
    ``{output_dir}/logs/{dataset}_{timestamp}.jsonl``.

    Args:
        output_dir: The run's per-key output directory (alongside the Excel
            files). The ``logs/`` subdirectory is created here.
        dataset: Dataset stem, used in the JSONL filename.
        timestamp: Run timestamp (reuse the one ``TBLSResultSaver`` already
            generates per run, so the JSONL and Excel share a timestamp).
    """
    logger.remove()
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable stdout sink (keep the current "{level} {message}" style).
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<level>{level: <8}</level> {message}",
    )

    # Structured JSONL file sink (one JSON object per line).
    logger.add(
        logs_dir / f"{dataset}_{timestamp}.jsonl",
        level="DEBUG",
        serialize=True,
        enqueue=False,
    )

    # Forward stdlib logging (evaluate.py warnings, etc.) into the same sinks.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
