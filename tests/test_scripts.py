"""The command-line entry points of M9 and M10 run end to end."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_prepare_examples_reports_missing_recordings(tmp_path: Path) -> None:
    run = _run("scripts/prepare_examples.py", "--recordings", str(tmp_path), "--out", str(tmp_path))
    assert run.returncode == 1
    assert "no usable recordings" in run.stderr


def test_make_report_writes_the_tables() -> None:
    before = {p.name: p.read_text() for p in (ROOT / "report" / "generated").glob("*.tex")}
    run = _run("scripts/make_report.py", "--no-pdf")
    assert run.returncode == 0, run.stderr
    after = {p.name: p.read_text() for p in (ROOT / "report" / "generated").glob("*.tex")}
    assert before == after


def test_demo_script_serves_the_interface() -> None:
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "scripts/demo.py", "live", "--no-browser", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        info = None
        for _ in range(100):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/info", timeout=2
                ) as reply:
                    info = json.loads(reply.read())
                break
            except OSError:
                time.sleep(0.1)
        assert info is not None and info["delay_samples"] == 511
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as page:
            assert b"HushWave" in page.read()
    finally:
        process.terminate()
        process.communicate(timeout=10)


def test_demo_script_reports_a_busy_port() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        run = _run("scripts/demo.py", "live", "--no-browser", "--port", str(port))
    assert run.returncode == 1
    assert "cannot listen" in run.stderr
