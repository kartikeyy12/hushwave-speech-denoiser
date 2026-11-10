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


def _speechlike(rng: np.random.Generator, n: int) -> np.ndarray:
    t = np.arange(n) / 16000
    x = 0.01 * rng.standard_normal(n)
    for start, stop in ((1.0, 1.8), (2.4, 3.3)):
        m = (t >= start) & (t < stop)
        x[m] += 0.2 * np.sin(2 * np.pi * 300 * t[m]) * np.hanning(int(m.sum()))
    return x


def test_recording_analysis_matches_python(tmp_path: Path, rng: np.random.Generator) -> None:
    from hushwave.nonintrusive import nonintrusive_metrics, segment_by_noise_floor

    n = 4 * 16000 + 100
    y = _speechlike(rng, n).astype("<f4")
    padded = np.pad(y, (0, (-n) % 256))
    z = _reference(padded)[:n].astype("<f4")
    source, enhanced, out = tmp_path / "y.f32", tmp_path / "z.f32", tmp_path / "out.json"
    y.tofile(source)
    z.tofile(enhanced)
    run = _node(JS / "analysis_check.js", source, enhanced, out)
    assert run.returncode == 0, run.stderr
    got = json.loads(out.read_text(encoding="utf-8"))

    yd, zd = y.astype(np.float64), z.astype(np.float64)
    seg = segment_by_noise_floor(yd, None)
    expected = nonintrusive_metrics(yd, zd, seg)
    assert seg.pause.sum() > 50 and seg.speech.sum() > 20
    assert got["pause"] == seg.pause.astype(int).tolist()
    assert got["speech"] == seg.speech.astype(int).tolist()
    m = got["metrics"]
    assert m["floorDb"] == pytest.approx(seg.floor_db, abs=1e-9)
    assert m["noiseReductionPauseDb"] == pytest.approx(
        expected["noise_reduction_pause_db"], abs=1e-6
    )
    assert m["speechLevelChangeDb"] == pytest.approx(expected["speech_level_change_db"], abs=1e-6)

    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(512) / 512)
    frames = (got["frameCount"] - 1) * 256 + 512
    assert frames <= n
    power = np.array(
        [
            np.abs(np.fft.rfft(yd[f * 256 : f * 256 + 512] * window)) ** 2 / np.sum(window**2)
            for f in range(got["frameCount"])
        ]
    )
    for pick, row in zip(got["picks"], got["frames"], strict=True):
        np.testing.assert_allclose(row, 10 * np.log10(power[pick] + 1e-12), atol=1e-3)
    np.testing.assert_allclose(
        got["averageAll"], 10 * np.log10(power.mean(axis=0) + 1e-12), atol=1e-3
    )
    mask = np.array(got["pauseFrames"], dtype=bool)
    assert mask.any()
    np.testing.assert_allclose(
        got["averagePause"], 10 * np.log10(power[mask].mean(axis=0) + 1e-12), atol=1e-3
    )
    energy = 10 * np.log10(np.mean(yd[: (n // 256) * 256].reshape(-1, 256) ** 2, axis=1) + 1e-20)
    np.testing.assert_allclose(got["energy"], energy, atol=1e-9)


def test_whole_recording_with_a_stand_in_engine() -> None:
    run = _node(JS / "whole_test.js")
    assert run.returncode == 0, run.stderr
    assert "whole: ok" in run.stdout


def test_charts_draw_without_errors() -> None:
    run = _node(JS / "charts_test.js")
    assert run.returncode == 0, run.stderr
    assert "charts: ok" in run.stdout


def test_whole_recording_through_the_firmware(
    host: Path, tmp_path: Path, rng: np.random.Generator
) -> None:
    n = 256 * 30 + 100
    x = _speechlike(rng, n)
    source, target, stats = tmp_path / "in.f32", tmp_path / "out.f32", tmp_path / "stats.json"
    x.astype("<f4").tofile(source)
    run = _node(JS / "whole_engine.js", host, source, target, stats)
    assert run.returncode == 0, run.stderr
    report = json.loads(stats.read_text(encoding="utf-8"))
    blocks = -(-(n + 511) // 256)
    assert report["blocks"] == report["received"] == blocks
    assert report["lost"] == 0 and report["dropped"] == 0
    assert report["computeCount"] == report["rttCount"] == blocks
    assert report["lastProgress"] == [blocks, blocks] and report["length"] == n
    out = np.fromfile(target, dtype="<f4")
    expected = _reference(np.pad(x, (0, blocks * 256 - n)))[511 : 511 + n]
    assert np.max(np.abs(out - expected)) < 4e-5


def test_wav_encoder(tmp_path: Path, rng: np.random.Generator) -> None:
    import wave

    x = np.clip(0.4 * rng.standard_normal(1000), -1.2, 1.2).astype("<f4")
    source, target = tmp_path / "x.f32", tmp_path / "x.wav"
    x.tofile(source)
    run = _node(JS / "wav_write.js", source, target, "16000")
    assert run.returncode == 0, run.stderr
    with wave.open(str(target), "rb") as reader:
        assert (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) == (
            1,
            2,
            16000,
        )
        pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    expected = np.clip(np.floor(x.astype(np.float64) * 32768 + 0.5), -32768, 32767)
    np.testing.assert_array_equal(pcm, expected.astype(np.int16))
