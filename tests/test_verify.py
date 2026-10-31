"""The quality gate itself must fail when it should (including the coverage threshold)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_verify() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_script", ROOT / "scripts" / "verify.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_script"] = module
    spec.loader.exec_module(module)
    return module


verify = _load_verify()


def test_run_gate_reports_pass_and_fail() -> None:
    passing = verify.run_gate("ok", [sys.executable, "-c", "print('fine')"])
    failing = verify.run_gate("bad", [sys.executable, "-c", "raise SystemExit(3)"])
    assert passing.status == "PASS"
    assert passing.detail == "fine"
    assert failing.status == "FAIL"


def test_run_gate_handles_empty_output() -> None:
    outcome = verify.run_gate("quiet", [sys.executable, "-c", "pass"])
    assert outcome.status == "PASS"
    assert outcome.detail == "(no output)"


def test_results_file_states(tmp_path: Path) -> None:
    assert verify.check_results_file(tmp_path).status == "N/A"
    (tmp_path / "results").mkdir()
    target = tmp_path / "results" / "results.json"
    target.write_text("{not json", encoding="utf-8")
    assert verify.check_results_file(tmp_path).status == "FAIL"
    target.write_text("[1, 2]", encoding="utf-8")
    assert verify.check_results_file(tmp_path).status == "FAIL"
    target.write_text('{"a": 1}', encoding="utf-8")
    outcome = verify.check_results_file(tmp_path)
    assert outcome.status == "PASS"
    assert "1 top-level" in outcome.detail


def test_coverage_threshold_actually_fails_the_gate(tmp_path: Path) -> None:
    """Run a thin slice of the suite with an impossible threshold: the gate must say FAIL."""
    environment = dict(os.environ, COVERAGE_FILE=str(tmp_path / ".coverage"))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_scaffold.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--cov=hushwave",
            "--cov-fail-under=100",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode != 0
    assert "Required test coverage of 100% not reached" in completed.stdout + completed.stderr


def test_coverage_line_is_summarised() -> None:
    script = "print('Total coverage: 97.50%'); print('=== 3 passed in 0.1s ===')"
    outcome = verify.run_gate("cov", [sys.executable, "-c", script])
    assert outcome.detail == "3 passed in 0.1s; Total coverage: 97.50%"


@pytest.mark.parametrize("name", ["lint (ruff check)", "format (ruff format)"])
def test_main_lists_expected_gates(name: str) -> None:
    source = (ROOT / "scripts" / "verify.py").read_text(encoding="utf-8")
    assert name in source
