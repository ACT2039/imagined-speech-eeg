"""Structured logging configuration for the imagined_speech package."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Configure and return the package root logger.

    Parameters
    ----------
    level : str
        Logging verbosity level (e.g., ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
    log_file : Path | str | None
        Optional path to a log file where messages will also be written.

    Returns
    -------
    logging.Logger
        Configured logger instance for ``imagined_speech``.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")

    logger = logging.getLogger("imagined_speech")
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Avoid attaching duplicate handlers on repeated invocations
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
