"""Shared pytest fixtures: every random test uses a fixed seed so runs are deterministic."""

from __future__ import annotations

import numpy as np
import pytest

SEED = 20260101


@pytest.fixture
def rng() -> np.random.Generator:
    """A freshly seeded generator; identical sequence in every test run."""
    return np.random.default_rng(SEED)
