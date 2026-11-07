"""Firmware parameters, serial protocol and the firmware compared with the Python reference."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hushwave import mcu
from hushwave.live import ROOT, TUNING_PATH, build_engine, load_b5
from hushwave.mcu import (
    BLOCK,
    SYNC,
    TYPE_AUDIO_IN,
    TYPE_AUDIO_OUT,
    TYPE_HELLO,
    TYPE_RESET_ACK,
    Frame,
    FrameParser,
    HostDevice,
    crc16,
    decode_audio_reply,
    decode_hello,
    encode_frame,
    firmware_params,
    float_from_pcm,
    pcm_from_float,
)

needs_compiler = pytest.mark.skipif(mcu.compiler() is None, reason="no C++ compiler")
FIRMWARE = ROOT / "firmware" / "hushwave_esp32"


def _reference(x: np.ndarray) -> np.ndarray:
    quantised = np.minimum(np.round(np.clip(x, -1, 1) * 32768) / 32768, 32767 / 32768)
    engine = build_engine(load_b5())
    out = np.concatenate(
        [engine.process(quantised[i : i + BLOCK]) for i in range(0, x.size, BLOCK)]
    )
    return np.asarray(np.clip(out, -1.0, 32767 / 32768), dtype=np.float64)  # 16-bit saturation


def _signals(rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = BLOCK * 300
    t = np.arange(n) / 16_000
    impulses = np.zeros(n)
    impulses[::4000] = 0.9
    return {
        "silence": np.zeros(n),
        "quiet": 1e-4 * rng.standard_normal(n),
        "loud_noise": 0.9 * rng.uniform(-1, 1, n),
        "clipped_tone": np.clip(3 * np.sin(2 * np.pi * 440 * t), -1, 1),
        "impulses": impulses,
        "step": np.where(t > 2, 0.3, 0.0) + 0.01 * rng.standard_normal(n),
        "tone_in_noise": 0.3 * np.sin(2 * np.pi * 1000 * t) + 0.02 * rng.standard_normal(n),
    }


@pytest.fixture(scope="module")
def host(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if mcu.compiler() is None:
        pytest.skip("no C++ compiler")
    return mcu.build_host(tmp_path_factory.mktemp("firmware"))


# ------------------------------------------------------------------ parameters
def test_committed_parameters_are_generated_from_the_tuning_file() -> None:
    assert mcu.PARAMS_PATH.read_text(encoding="utf-8") == firmware_params()
    text = mcu.PARAMS_PATH.read_text(encoding="utf-8")
    assert (
        "kMcraDelta = 3.0f" in text
        and "kGainFloor = 0.03f" in text
        and "kHpCutoffHz = 80.0f" in text
    )


def test_write_params(tmp_path: Path) -> None:
    out = mcu.write_params(tmp_path / "p.h")
    assert out.read_text(encoding="utf-8") == firmware_params()


def _tuning_with(tmp_path: Path, **changes: object) -> Path:
    data = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
    data["final"]["b5"].update(changes)
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rule": "lsa"}, "Wiener, MCRA"),
        ({"estimator": "minstats"}, "Wiener, MCRA"),
        ({"prefilter": "biquad"}, "Wiener, MCRA"),
        ({"est": [["delta", 3.0]]}, "rise limit"),
        (
            {"gain": [["g_min", 0.03], ["dd_alpha", 0.9], ["spectral_hp", 1.0], ["gain_mu", 0.5]]},
            "no gain smoothing",
        ),
    ],
)
def test_parameters_refuse_settings_the_firmware_cannot_run(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        firmware_params(_tuning_with(tmp_path, **changes))


def test_parameters_refuse_other_frame_sizes(tmp_path: Path) -> None:
    path = _tuning_with(tmp_path, frame_ms=16)
    with pytest.raises(ValueError, match="16 kHz"):
        firmware_params(path)


# -------------------------------------------------------------------- protocol
def test_crc16_check_value() -> None:
    assert crc16(b"123456789") == 0x29B1
    assert crc16(b"") == 0xFFFF


def test_frame_round_trip_and_layout() -> None:
    payload = bytes(range(200))
    frame = encode_frame(TYPE_AUDIO_IN, 77, payload)
    assert frame[:6] == SYNC + bytes([TYPE_AUDIO_IN, 77]) + struct.pack("<H", 200)
    assert len(frame) == 8 + 200
    assert list(FrameParser().push(frame)) == [Frame(TYPE_AUDIO_IN, 77, payload)]


def test_encode_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="one byte"):
        encode_frame(300, 0)
    with pytest.raises(ValueError, match="one byte"):
        encode_frame(1, 256)
    with pytest.raises(ValueError, match="longer"):
        encode_frame(1, 0, bytes(mcu.MAX_PAYLOAD + 1))


def test_parser_handles_split_garbage_and_damaged_frames() -> None:
    frame = encode_frame(TYPE_AUDIO_IN, 1, bytes(512))
    parser = FrameParser()
    found: list[Frame] = []
    for i in range(0, len(frame), 7):
        found.extend(parser.push(frame[i : i + 7]))
    assert len(found) == 1
    parser = FrameParser()
    damaged = bytearray(frame)
    damaged[300] ^= 0x01
    stream = b"\x00\x01\xa5" + bytes(damaged) + b"\xa5\xa5\x03" + frame + b"\xa5"
    assert [f.seq for f in parser.push(stream)] == [1]
    assert parser.crc_errors >= 1 and parser.skipped_bytes >= 5


def test_parser_rejects_impossible_lengths() -> None:
    frame = encode_frame(TYPE_HELLO, 4)
    parser = FrameParser()
    out = list(parser.push(SYNC + bytes([1, 1]) + b"\xff\xff" + frame))
    assert [f.seq for f in out] == [4] and parser.length_errors >= 1


def test_pcm_conversion_matches_the_firmware_rules() -> None:
    pcm = pcm_from_float([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 1 / 32768])
    assert list(np.frombuffer(pcm, dtype="<i2")) == [
        0,
        16384,
        -16384,
        32767,
        -32768,
        32767,
        -32768,
        1,
    ]
    np.testing.assert_allclose(float_from_pcm(pcm)[:3], [0.0, 0.5, -0.5])


def test_reply_decoders_reject_wrong_frames() -> None:
    with pytest.raises(ValueError, match="audio reply"):
        decode_audio_reply(Frame(TYPE_AUDIO_OUT, 1, b"short"))
    with pytest.raises(ValueError, match="hello"):
        decode_hello(Frame(0x82, 1, b"x"))
    reply = decode_audio_reply(
        Frame(TYPE_AUDIO_OUT, 5, struct.pack("<HH", 123, 0) + bytes(2 * BLOCK))
    )
    assert (reply.seq, reply.compute_us, reply.status, reply.samples.size) == (5, 123, 0, BLOCK)


# ------------------------------------------------------------- firmware on a PC
@needs_compiler
def test_hello_describes_the_firmware(host: Path) -> None:
    with HostDevice(host) as device:
        hello = device.hello()
    assert (hello.version, hello.fs, hello.block, hello.delay) == (1, 16_000, 256, 511)
    assert "HushWave" in hello.name


@needs_compiler
@pytest.mark.parametrize("name", list(_signals(np.random.default_rng(0))))
def test_firmware_matches_the_python_reference(host: Path, name: str) -> None:
    x = _signals(np.random.default_rng(5))[name]
    with HostDevice(host) as device:
        out, times = device.process(x)
    difference = out - _reference(x)
    assert np.all(np.isfinite(out))
    # float32 against float64, plus rounding of the output to 16 bit (half a step is 1.5e-5)
    assert np.max(np.abs(difference)) < 4e-5
    assert np.sqrt(np.mean(difference**2)) < 1.5e-5
    assert all(0 <= t <= 0xFFFF for t in times)


@needs_compiler
def test_pipelining_depth_does_not_change_the_result(host: Path, rng: np.random.Generator) -> None:
    x = 0.05 * rng.standard_normal(BLOCK * 60)
    results = []
    for depth in (1, 4, 8):
        with HostDevice(host) as device:
            results.append(device.process(x, depth=depth)[0])
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])


@needs_compiler
def test_reset_restarts_the_stream(host: Path, rng: np.random.Generator) -> None:
    x = 0.05 * rng.standard_normal(BLOCK * 40)
    with HostDevice(host) as device:
        first, _ = device.process(x)
        device.process(0.5 * rng.standard_normal(BLOCK * 10))
        device.reset()
        second, _ = device.process(x)
    np.testing.assert_array_equal(first, second)


@needs_compiler
def test_the_stream_delay_is_the_documented_one(host: Path) -> None:
    x = np.zeros(BLOCK * 20)
    x[BLOCK * 2] = 0.5
    with HostDevice(host) as device:
        out, _ = device.process(x)
    assert abs(int(np.argmax(np.abs(out))) - (BLOCK * 2 + 511)) <= 2


@needs_compiler
def test_the_firmware_survives_bad_input(host: Path, rng: np.random.Generator) -> None:
    block = pcm_from_float(0.05 * rng.standard_normal(BLOCK))
    good = encode_frame(TYPE_AUDIO_IN, 9, block)
    bad_crc = bytearray(good)
    bad_crc[100] ^= 0xFF
    short_audio = encode_frame(TYPE_AUDIO_IN, 10, bytes(100))
    unknown = encode_frame(0x55, 11, b"abc")
    with HostDevice(host) as device:
        device.write(b"\x00\xa5\x01" + bytes(bad_crc) + short_audio + unknown + b"junk" + good)
        frames = device.read_frames(1)
        assert [f.seq for f in frames] == [9] and frames[0].kind == TYPE_AUDIO_OUT
        reply = device.request(TYPE_HELLO)
        assert reply.kind == 0x82
        assert device.request(0x03).kind == TYPE_RESET_ACK


@needs_compiler
def test_process_needs_whole_blocks(host: Path) -> None:
    with HostDevice(host) as device, pytest.raises(ValueError, match="multiple of"):
        device.process(np.zeros(BLOCK + 1))


@needs_compiler
def test_host_device_reports_a_dead_process(host: Path) -> None:
    device = HostDevice(host)
    device.close()
    with pytest.raises((EOFError, BrokenPipeError, ValueError, OSError)):
        device.request(TYPE_HELLO)
    with pytest.raises(RuntimeError, match="acknowledgement"):
        reply_device = HostDevice(host)
        try:
            reply_device.request = lambda *_args, **_kw: Frame(0, 0, b"")  # type: ignore[method-assign]
            reply_device.reset()
        finally:
            reply_device.close()


@needs_compiler
def test_build_failure_is_reported(tmp_path: Path) -> None:
    broken = tmp_path / "src"
    broken.mkdir()
    (broken / "host_main.cpp").write_text("int main() { return undefined_symbol; }\n")
    with pytest.raises(RuntimeError, match="build failed"):
        mcu.build_host(tmp_path / "out", source_dir=broken)


def test_build_needs_a_compiler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="build-essential"):
        mcu.build_host(tmp_path)


@needs_compiler
def test_the_arduino_sketch_compiles_against_a_mock_board() -> None:
    cxx = mcu.compiler()
    assert cxx is not None
    command = [
        cxx, "-std=c++11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", "-x", "c++",
        "-include", str(ROOT / "firmware" / "host" / "arduino_mock" / "Arduino.h"),
        f"-I{FIRMWARE}", str(FIRMWARE / "hushwave_esp32.ino"),
    ]  # fmt: skip
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    base = [cxx, "-std=c++11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", "-DARDUINO=10819"]
    for name in mcu.SOURCES:
        syntax = subprocess.run(
            [*base, f"-I{FIRMWARE}", str(FIRMWARE / name)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr


def test_the_sketch_folder_follows_the_arduino_naming_rule() -> None:
    assert (FIRMWARE / "hushwave_esp32.ino").is_file()
    text = (FIRMWARE / "hushwave_esp32.ino").read_text(encoding="utf-8")
    assert "Serial.setRxBufferSize(4096)" in text and text.index("setRxBufferSize") < text.index(
        "Serial.begin"
    )


@needs_compiler
def test_firmware_check_report(host: Path) -> None:
    report = mcu.check_firmware(host)
    assert report["protocol_version"] == 1 and report["delay_samples"] == 511
    rows = report["signals"]
    assert isinstance(rows, list) and len(rows) == 7
    assert all(r["finite"] for r in rows)
    assert float(report["max_abs_difference"]) < 2 * float(report["pcm_step"])  # type: ignore[arg-type]
    assert 20_000 < float(report["engine_bytes"]) < 60_000  # type: ignore[arg-type]
    assert float(report["uplink_bytes_per_second"]) < 0.5 * float(report["baud_bytes_per_second"])  # type: ignore[arg-type]


@needs_compiler
def test_check_script_is_reproducible() -> None:
    path = ROOT / "results" / "firmware_check.json"
    before = path.read_text(encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_firmware.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert path.read_text(encoding="utf-8") == before


def test_params_script_writes_the_committed_header() -> None:
    before = mcu.PARAMS_PATH.read_text(encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_firmware_params.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert mcu.PARAMS_PATH.read_text(encoding="utf-8") == before
