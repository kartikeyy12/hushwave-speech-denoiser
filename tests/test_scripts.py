"""The command-line entry points of M9 and M10 run end to end."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import ModuleType

import pytest

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


def _demo_module() -> ModuleType:
    import importlib.util

    spec = importlib.util.spec_from_file_location("demo_script", ROOT / "scripts" / "demo.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wsl_detection(tmp_path: Path) -> None:
    demo = _demo_module()
    wsl = tmp_path / "version"
    wsl.write_text("Linux version 5.15 (Microsoft-standard-WSL2)", encoding="utf-8")
    plain = tmp_path / "plain"
    plain.write_text("Linux version 6.8 generic", encoding="utf-8")
    assert demo.in_wsl(wsl) is True
    assert demo.in_wsl(plain) is False
    assert demo.in_wsl(tmp_path / "missing") is False


def test_browser_opens_through_windows_under_wsl(monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _demo_module()
    calls: list[object] = []
    monkeypatch.setattr(demo, "in_wsl", lambda: True)
    monkeypatch.setattr(demo.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    monkeypatch.setattr(demo.webbrowser, "open", lambda url: calls.append(("web", url)))
    demo.open_browser("http://localhost:1/#live")
    assert calls == [["explorer.exe", "http://localhost:1/#live"]]


def test_browser_falls_back_to_the_standard_library(monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _demo_module()
    calls: list[object] = []
    monkeypatch.setattr(demo, "in_wsl", lambda: False)
    monkeypatch.setattr(demo.webbrowser, "open", lambda url: calls.append(url))
    demo.open_browser("http://localhost:1/#live")
    assert calls == ["http://localhost:1/#live"]

    def broken(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("explorer.exe")

    monkeypatch.setattr(demo, "in_wsl", lambda: True)
    monkeypatch.setattr(demo.subprocess, "run", broken)
    demo.open_browser("http://localhost:2/#live")
    assert calls[-1] == "http://localhost:2/#live"
