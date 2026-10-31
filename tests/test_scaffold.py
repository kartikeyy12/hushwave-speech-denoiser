"""Checks on the repository scaffold itself (milestone M0)."""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

import numpy as np

import hushwave

ROOT = Path(__file__).resolve().parents[1]
# Built from pieces so that this file never matches its own scan.
MARKER_PATTERN = re.compile("|".join(("TO" + "DO", "FIX" + "ME", "NotImplemented" + "Error")))


def test_version_matches_installed_metadata() -> None:
    assert hushwave.__version__ == version("hushwave")


def test_required_top_level_files_exist() -> None:
    for name in ("README.md", "LICENSE", "Makefile", "pyproject.toml", ".gitignore"):
        assert (ROOT / name).is_file(), name


def test_required_directories_exist() -> None:
    for name in ("src/hushwave", "scripts", "tests", "data", "results", "docs", "report"):
        assert (ROOT / name).is_dir(), name


def test_rng_fixture_is_deterministic(rng: np.random.Generator) -> None:
    again = np.random.default_rng(20260101)
    np.testing.assert_array_equal(rng.standard_normal(8), again.standard_normal(8))


def test_no_placeholder_markers_in_code() -> None:
    offenders: list[str] = []
    for folder in ("src", "scripts", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if MARKER_PATTERN.search(text):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
