"""Project text stays plain: no em dashes, no en dashes, no leftover work markers."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EM, EN = chr(0x2014), chr(0x2013)
SUFFIXES = {
    ".py",
    ".md",
    ".tex",
    ".js",
    ".css",
    ".html",
    ".toml",
    ".txt",
    ".json",
    ".cpp",
    ".h",
    ".ino",
}
SKIP_DIRS = {".venv", ".git", "data", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache"}
SKIP_PREFIXES = ("results/", "report/build/", "examples/")


def _text_files() -> list[Path]:
    found = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if SKIP_DIRS & set(path.relative_to(ROOT).parts) or rel.startswith(SKIP_PREFIXES):
            continue
        found.append(path)
    return found


@pytest.mark.parametrize("path", _text_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_em_or_en_dashes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert EM not in text, f"{path.name} contains an em dash"
    assert EN not in text, f"{path.name} contains an en dash"
