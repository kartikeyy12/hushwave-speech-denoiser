"""scripts/verify.py records the test count and coverage of a passing run for the report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _verify() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_script", ROOT / "scripts" / "verify.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_is_written(tmp_path: Path) -> None:
    verify = _verify()
    (tmp_path / "results").mkdir()
    log = "...\n850 passed in 120.00s\nTotal coverage: 98.53%\n"
    outcomes = [
        verify.Outcome("lint (ruff check)", "PASS", "", ""),
        verify.Outcome("tests + coverage >= 90%", "PASS", "", log),
    ]
    verify.write_summary(outcomes, tmp_path)
    data = json.loads((tmp_path / "results" / "verification.json").read_text(encoding="utf-8"))
    assert data == {"tests_passed": 850, "coverage_percent": 98.53}


def test_summary_needs_a_test_gate_with_numbers(tmp_path: Path) -> None:
    verify = _verify()
    (tmp_path / "results").mkdir()
    verify.write_summary([verify.Outcome("lint", "PASS", "", "")], tmp_path)
    verify.write_summary([verify.Outcome("tests", "PASS", "", "no numbers here")], tmp_path)
    assert not (tmp_path / "results" / "verification.json").exists()
