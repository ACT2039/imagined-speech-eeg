"""Reproducibility and deterministic seeding utilities (Rule 8)."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

from imagined_speech.exceptions import ConfigurationError

logger = logging.getLogger("imagined_speech.utils.reproducibility")


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed standard library and NumPy random number generators for reproducibility.

    Parameters
    ----------
    seed : int
        Non-negative integer random seed.

    Returns
    -------
    np.random.Generator
        A freshly seeded ``numpy.random.Generator`` instance for explicit passing.

    Raises
    ------
    ConfigurationError
        If ``seed`` is not a non-negative integer.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigurationError(
            f"Random seed must be a non-negative integer, got {seed!r}."
        )

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    logger.debug("Global random seed configured to %d.", seed)
    return rng
