"""Run every quality gate and print ONE PASS/FAIL table.

Gates: lint (ruff), format (ruff), type-check (mypy), tests with coverage (>= 90 %),
and the results-file check. Exit status is 0 only when no gate reports FAIL.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE_MINIMUM = 90


@dataclass(frozen=True)
class Outcome:
    name: str
    status: str
    detail: str
    log: str


def run_gate(name: str, command: list[str]) -> Outcome:
    """Run one external tool from the repository root and summarise its result."""
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    output = (completed.stdout + completed.stderr).strip()
    lines = [line for line in output.splitlines() if line.strip()]
    detail = lines[-1].strip("= ") if lines else "(no output)"
    coverage = re.search(r"Total coverage: [\d.]+%", output)
    if coverage:
        detail = f"{detail}; {coverage.group(0)}"
    status = "PASS" if completed.returncode == 0 else "FAIL"
    return Outcome(name, status, detail, output)


def check_results_file(root: Path = ROOT) -> Outcome:
    """The results file is created in M8; once present it must be a JSON object."""
    path = root / "results" / "results.json"
    if not path.exists():
        return Outcome(
            "results schema", "N/A", "results/results.json is created in milestone M8", ""
        )
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return Outcome("results schema", "FAIL", f"cannot parse results.json: {error}", "")
    if not isinstance(parsed, dict):
        return Outcome("results schema", "FAIL", "results.json is not a JSON object", "")
    return Outcome("results schema", "PASS", f"{len(parsed)} top-level keys", "")


def main() -> int:
    python = sys.executable
    outcomes = [
        run_gate("lint (ruff check)", [python, "-m", "ruff", "check", "src", "scripts", "tests"]),
        run_gate(
            "format (ruff format)",
            [python, "-m", "ruff", "format", "--check", "src", "scripts", "tests"],
        ),
        run_gate("type-check (mypy)", [python, "-m", "mypy"]),
        run_gate(
            f"tests + coverage >= {COVERAGE_MINIMUM}%",
            [
                python,
                "-m",
                "pytest",
                "--cov=hushwave",
                "--cov-report=term-missing",
                f"--cov-fail-under={COVERAGE_MINIMUM}",
            ],
        ),
        check_results_file(),
    ]

    width = max(len(outcome.name) for outcome in outcomes)
    print(f"{'GATE'.ljust(width)}  STATUS  DETAIL")
    print(f"{'-' * width}  ------  ------")
    for outcome in outcomes:
        print(f"{outcome.name.ljust(width)}  {outcome.status.ljust(6)}  {outcome.detail}")

    failed = [outcome for outcome in outcomes if outcome.status == "FAIL"]
    for outcome in failed:
        print(f"\n--- full output of failed gate: {outcome.name} ---")
        print(outcome.log)

    overall = "FAIL" if failed else "PASS"
    print(f"\nOVERALL: {overall}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
