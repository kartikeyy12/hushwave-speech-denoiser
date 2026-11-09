"""The stand-alone board probe (scripts/mcu_probe.py) against the real firmware and stand-ins."""

from __future__ import annotations

import importlib.util
import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from hushwave import mcu
from hushwave.live import ROOT

needs_compiler = pytest.mark.skipif(mcu.compiler() is None, reason="no C++ compiler")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mcu_probe", ROOT / "scripts" / "mcu_probe.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcu_probe"] = module
    spec.loader.exec_module(module)
    return module


probe = _load()


class PipeLink:
    """The firmware executable behind two pipes, shaped like a serial port."""

    def __init__(self, executable: Path) -> None:
        self._process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self._chunks: queue.Queue[bytes] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self._process.stdout is not None
        while chunk := self._process.stdout.read1(65536):  # type: ignore[attr-defined]
            self._chunks.put(chunk)

    def write(self, data: bytes) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(data)
        self._process.stdin.flush()

    def read(self, timeout: float) -> bytes:
        try:
            return self._chunks.get(timeout=timeout)
        except queue.Empty:
            return b""

    def close(self) -> None:
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.close()
        self._process.wait(timeout=10)
        self._process.stdout.close()


class QuietLink:
    """A port that opens but says only what it was given."""

    def __init__(self, text: bytes = b"") -> None:
        self._text = text

    def write(self, data: bytes) -> None:
        pass

    def read(self, timeout: float) -> bytes:
        text, self._text = self._text, b""
        if not text:
            probe.time.sleep(min(timeout, 0.01))
        return bytes(text)

    def close(self) -> None:
        pass


@pytest.fixture(scope="module")
def host(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if mcu.compiler() is None:
        pytest.skip("no C++ compiler")
    return mcu.build_host(tmp_path_factory.mktemp("firmware"))


def test_the_copy_of_the_protocol_matches_the_project(rng: np.random.Generator) -> None:
    assert probe.crc16(b"123456789") == 0x29B1
    data = rng.integers(0, 256, 300, dtype=np.uint8).tobytes()
    assert probe.crc16(data) == mcu.crc16(data)
    payload = data[:200]
    assert probe.encode_frame(0x01, 7, payload) == mcu.encode_frame(0x01, 7, payload)


def test_the_parser_resynchronises_after_noise_and_damage() -> None:
    good = probe.encode_frame(0x82, 3, b"abc")
    damaged = bytearray(probe.encode_frame(0x81, 4, b"xyz"))
    damaged[-1] ^= 0xFF
    parser = probe.Parser()
    frames = []
    for chunk in (b"boot text\r\n" + bytes(damaged)[:5], bytes(damaged)[5:] + good[:4], good[4:]):
        frames += parser.push(chunk)
    assert [(f.kind, f.seq, f.payload) for f in frames] == [(0x82, 3, b"abc")]
    assert parser.crc_errors == 1
    assert parser.stray.startswith(b"boot text")


@needs_compiler
def test_probe_passes_against_the_firmware(host: Path, capsys: pytest.CaptureFixture[str]) -> None:
    link = PipeLink(host)
    try:
        report = probe.probe(link, blocks=60, wait=3.0, realtime=False)
    finally:
        link.close()
    assert report.ok and report.sent == report.received == 60 and report.lost == 0
    assert report.hello is not None and report.hello.name == "HushWave B5 float32"
    assert report.hello.block == 256 and report.hello.delay == 511
    probe.print_report(report)
    out = capsys.readouterr().out
    assert "result: PASS" in out and "round trip" in out and "compute" in out


def test_a_silent_port_is_explained(capsys: pytest.CaptureFixture[str]) -> None:
    report = probe.probe(QuietLink(), blocks=5, wait=0.4)
    assert not report.ok and report.hello is None
    probe.print_report(report)
    out = capsys.readouterr().out
    assert "did not answer" in out and "USB CDC On Boot" in out and "result: FAIL" in out


def test_a_port_that_prints_text_is_explained(capsys: pytest.CaptureFixture[str]) -> None:
    report = probe.probe(QuietLink(b"ESP-ROM:esp32s3-20210327\r\n"), blocks=5, wait=0.4)
    probe.print_report(report)
    out = capsys.readouterr().out
    assert "not HushWave frames" in out and "ESP-ROM" in out


def test_port_names_and_hints() -> None:
    board = probe.PortInfo("COM10", "USB Serial Device (COM10)", 0x303A, 0x1001)
    bridge = probe.PortInfo("COM3", "Silicon Labs CP210x", 0x10C4, 0xEA60)
    other = probe.PortInfo("COM1", "Communications Port", None, None)
    assert "native USB" in probe.format_port(board) and "303A:1001" in probe.format_port(board)
    assert "CP210x" in probe.format_port(bridge)
    assert probe.format_port(other).endswith("[USB -]")
    assert probe.choose_port([other, board, bridge]) == "COM10"
    assert probe.choose_port([board, board]) is None
    assert "holds the port" in probe.open_failure_hint("PermissionError(13, 'Access is denied.')")
    assert "port is gone" in probe.open_failure_hint("FileNotFoundError could not open port")
    assert "replug" in probe.open_failure_hint("something else")


def test_watch_reports_appearing_and_disappearing_ports(
    capsys: pytest.CaptureFixture[str],
) -> None:
    board = probe.PortInfo("COM10", "USB Serial Device", 0x303A, 0x1001)
    states = iter([[], [board], [board], []])
    last: list[object] = []

    def lister() -> list[object]:
        nonlocal last
        last = next(states, last)
        return last

    probe.watch_ports(0.75, lister=lister, sleep=lambda seconds: None)
    out = capsys.readouterr().out
    assert "appeared" in out and "gone" in out


def test_main_lists_ports_and_reports_a_busy_port(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    board = probe.PortInfo("COM10", "USB Serial Device", 0x303A, 0x1001)
    monkeypatch.setattr(probe, "list_ports", lambda: [board])
    assert probe.main(["--list"]) == 0
    assert "COM10" in capsys.readouterr().out

    def refuse(port: str, baud: int, dtr: bool, rts: bool) -> object:
        raise PermissionError(13, "Access is denied.")

    monkeypatch.setattr(probe, "open_link", refuse)
    assert probe.main([]) == 2
    out = capsys.readouterr().out
    assert "Using COM10" in out and "holds the port" in out and "result: FAIL" in out


def test_main_runs_a_probe_on_the_chosen_port(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[tuple[str, int, bool, bool]] = []

    def quiet(port: str, baud: int, dtr: bool, rts: bool) -> QuietLink:
        seen.append((port, baud, dtr, rts))
        return QuietLink()

    monkeypatch.setattr(probe, "open_link", quiet)
    code = probe.main(["--port", "COM9", "--settle", "0", "--wait", "0.3", "--dtr", "on"])
    assert code == 1 and seen == [("COM9", 921600, True, False)]
    assert "result: FAIL" in capsys.readouterr().out
