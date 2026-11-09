"""The page's JavaScript against the real firmware and the real server, through Node."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hushwave import mcu
from hushwave.demo_server import BLOCK, DemoApp, make_server
from hushwave.live import ROOT, LiveSession, build_engine, load_b5

JS = ROOT / "tests" / "js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@pytest.fixture(scope="module")
def host(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if mcu.compiler() is None:
        pytest.skip("no C++ compiler")
    return mcu.build_host(tmp_path_factory.mktemp("firmware"))


def _node(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", *map(str, args)], capture_output=True, text=True, timeout=120, check=False
    )


def _stream(
    host: Path, x: np.ndarray, tmp: Path, options: dict[str, object]
) -> tuple[np.ndarray, dict[str, Any]]:
    source, target, stats = tmp / "in.f32", tmp / "out.f32", tmp / "stats.json"
    x.astype("<f4").tofile(source)
    run = _node(JS / "engine_roundtrip.js", host, source, target, stats, json.dumps(options))
    assert run.returncode == 0, run.stderr
    return np.fromfile(target, dtype="<f4"), json.loads(stats.read_text(encoding="utf-8"))


def _reference(x: np.ndarray) -> np.ndarray:
    quantised = np.minimum(np.round(np.clip(x.astype("<f4"), -1, 1) * 32768) / 32768, 32767 / 32768)
    engine = build_engine(load_b5())
    out = np.concatenate([engine.process(quantised[i : i + 256]) for i in range(0, x.size, 256)])
    return np.asarray(np.clip(out, -1.0, 32767 / 32768), dtype=np.float64)


def test_protocol_codec() -> None:
    run = _node(JS / "protocol_test.js")
    assert run.returncode == 0, run.stderr
    assert "protocol: ok" in run.stdout


def test_error_wording() -> None:
    run = _node(JS / "errors_test.js")
    assert run.returncode == 0, run.stderr
    assert "errors: ok" in run.stdout


def test_open_retries_and_uses_the_run_state(
    host: Path, tmp_path: Path, rng: np.random.Generator
) -> None:
    x = 0.05 * rng.standard_normal(256 * 20)
    out, report = _stream(host, x, tmp_path, {"openFailures": 2})
    record = report["record"]
    assert record["openCalls"] == 3
    assert record["openOptions"][-1] == {
        "baudRate": 921600,
        "bufferSize": 65536,
        "flowControl": "none",
    }
    assert record["signals"] == [{"dataTerminalReady": False, "requestToSend": False}]
    assert report["received"] == 20
    assert np.max(np.abs(out - _reference(x))) < 4e-5


def test_a_port_that_never_opens_gives_the_browser_error(tmp_path: Path) -> None:
    source, stats = tmp_path / "in.f32", tmp_path / "stats.json"
    np.zeros(256, dtype="<f4").tofile(source)
    options = json.dumps({"openFailures": 99, "openAttempts": 3})
    run = _node(
        JS / "engine_roundtrip.js", "/bin/true", source, tmp_path / "out.f32", stats, options
    )
    assert run.returncode != 0
    report = json.loads(stats.read_text(encoding="utf-8"))
    assert "Failed to open serial port" in report["error"]
    assert report["record"]["openCalls"] == 3
    assert report["record"]["signals"] == []


def test_a_failed_handshake_releases_the_port(tmp_path: Path) -> None:
    silent = tmp_path / "silent.sh"
    silent.write_text("#!/bin/sh\ncat > /dev/null\n")
    silent.chmod(0o755)
    source, stats = tmp_path / "in.f32", tmp_path / "stats.json"
    np.zeros(256, dtype="<f4").tofile(source)
    options = json.dumps({"handshakeMs": 600})
    run = _node(JS / "engine_roundtrip.js", silent, source, tmp_path / "out.f32", stats, options)
    assert run.returncode != 0
    report = json.loads(stats.read_text(encoding="utf-8"))
    assert report["record"]["closed"] is True
    assert report["locked"] is False


def test_esp32_engine_matches_the_python_reference(
    host: Path, tmp_path: Path, rng: np.random.Generator
) -> None:
    x = 0.05 * rng.standard_normal(256 * 120)
    out, report = _stream(host, x, tmp_path, {})
    assert report["info"] == {
        "fs": 16000,
        "block": 256,
        "delay": 511,
        "name": "HushWave B5 float32",
    }
    assert report["received"] == report["blocks"] == 120
    assert report["stats"]["dropped"] == 0 and report["stats"]["lost"] == 0
    assert np.max(np.abs(out - _reference(x))) < 4e-5


def test_handshake_retries_until_the_board_answers(
    host: Path, tmp_path: Path, rng: np.random.Generator
) -> None:
    x = 0.05 * rng.standard_normal(256 * 20)
    out, report = _stream(host, x, tmp_path, {"dropWrites": [0, 1]})
    assert report["received"] == 20
    assert np.max(np.abs(out - _reference(x))) < 4e-5


def test_a_lost_or_damaged_frame_costs_one_block_and_nothing_else(
    host: Path, tmp_path: Path, rng: np.random.Generator
) -> None:
    x = 0.05 * rng.standard_normal(256 * 60)
    # write 0 is the hello, write 1 the reset, writes 2 and up are audio blocks 0, 1, ...
    options = {"dropWrites": [10], "corruptWrites": [30], "timeoutMs": 200}
    out, report = _stream(host, x, tmp_path, options)
    assert report["stats"]["lost"] == 2
    assert report["received"] == 58
    assert np.all(np.isfinite(out))


def test_a_silent_board_gives_a_clear_error(tmp_path: Path) -> None:
    silent = tmp_path / "silent.sh"
    silent.write_text("#!/bin/sh\ncat > /dev/null\n")
    silent.chmod(0o755)
    source = tmp_path / "in.f32"
    np.zeros(256, dtype="<f4").tofile(source)
    options = json.dumps({"handshakeMs": 600})
    run = _node(
        JS / "engine_roundtrip.js",
        silent,
        source,
        tmp_path / "out.f32",
        tmp_path / "stats.json",
        options,
    )
    assert run.returncode != 0
    assert "did not answer" in run.stderr


def test_python_engine_matches_the_session(tmp_path: Path, rng: np.random.Generator) -> None:
    app = DemoApp(load_b5())
    server = make_server(app, "127.0.0.1", 0)
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    try:
        x = (0.05 * rng.standard_normal(BLOCK * 40)).astype("<f4")
        source, target = tmp_path / "in.f32", tmp_path / "out.f32"
        x.tofile(source)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        run = _node(JS / "python_engine.js", base, source, target)
        assert run.returncode == 0, run.stderr
        report = json.loads(run.stdout)
        assert report["received"] == report["blocks"] == 40
        out = np.fromfile(target, dtype="<f4")
        session = LiveSession(load_b5())
        expected = np.concatenate(
            [
                session.process(x[i : i + BLOCK].astype(np.float64), True)[0]
                for i in range(0, x.size, BLOCK)
            ]
        )
        np.testing.assert_allclose(out, expected, atol=1e-6)
    finally:
        server.shutdown()
        server.server_close()
