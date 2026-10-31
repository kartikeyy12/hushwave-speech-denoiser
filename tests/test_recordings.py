"""Recording checks used on your own fan and corridor files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hushwave.recordings import analyze_recording, prepare_recording

FS = 16_000


def _noise(seconds: float, level: float, rng: np.random.Generator) -> np.ndarray:
    return level * rng.standard_normal(int(seconds * FS))


def test_good_recording_has_no_warnings(rng: np.random.Generator) -> None:
    report = analyze_recording(_noise(40.0, 0.05, rng), FS)
    assert report.warnings == ()
    assert report.duration_s == pytest.approx(40.0)
    assert report.rms_dbfs == pytest.approx(20 * np.log10(0.05), abs=0.1)


def test_each_problem_is_reported(rng: np.random.Generator) -> None:
    assert any("too short" in w for w in analyze_recording(_noise(5.0, 0.05, rng), FS).warnings)
    loud = np.clip(_noise(40.0, 1.0, rng), -1.0, 1.0)
    warnings = analyze_recording(loud, FS).warnings
    assert any("clipping" in w for w in warnings)
    assert any("full scale" in w for w in warnings)
    quiet = analyze_recording(_noise(40.0, 0.0005, rng), FS).warnings
    assert any("very quiet" in w for w in quiet)
    assert any(
        "DC offset" in w for w in analyze_recording(_noise(40.0, 0.01, rng) + 0.2, FS).warnings
    )
    gated = _noise(40.0, 0.05, rng)
    gated[: 20 * FS] = 0.0
    assert any("noise gate" in w for w in analyze_recording(gated, FS).warnings)


def test_short_inputs_and_bad_shapes() -> None:
    report = analyze_recording(np.full(100, 0.1), FS, min_duration_s=0.0)
    assert report.quiet_frame_dbfs == pytest.approx(20 * np.log10(0.1), abs=0.01)
    with pytest.raises(ValueError):
        analyze_recording(np.zeros((2, 2)), FS)
    with pytest.raises(ValueError):
        analyze_recording(np.zeros(0), FS)


def test_prepare_recording_converts_to_16k_mono_wav(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    source = tmp_path / "raw.wav"
    stereo = 0.05 * rng.standard_normal((32 * 48_000, 2))
    sf.write(str(source), stereo, 48_000, subtype="PCM_16")
    destination = tmp_path / "out" / "fan.wav"
    report = prepare_recording(source, destination)
    info = sf.info(str(destination))
    assert (info.samplerate, info.channels) == (16_000, 1)
    assert info.frames / info.samplerate == pytest.approx(32.0, abs=0.01)
    assert report.warnings == ()
