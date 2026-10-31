"""Audio I/O: mono, sample-rate policy, clipping, and PCM conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hushwave.audio_io import (
    float_to_pcm16,
    limit_peak,
    read_audio,
    resample,
    to_mono,
    write_wav,
)
from hushwave.fft import rfft


def test_write_read_round_trip_is_within_half_an_lsb(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    x = 0.5 * rng.uniform(-1.0, 1.0, 4000)
    path = tmp_path / "a.wav"
    assert write_wav(path, x) == 0
    y, fs = read_audio(path)
    assert fs == 16_000
    assert np.max(np.abs(y - x)) <= 0.5 / 32768.0 + 1e-12


def test_stereo_is_averaged_to_mono(tmp_path: Path) -> None:
    left, right = np.full(1000, 0.2), np.full(1000, 0.4)
    path = tmp_path / "s.wav"
    sf.write(str(path), np.stack([left, right], axis=1), 16_000, subtype="PCM_16")
    y, _ = read_audio(path)
    assert y.shape == (1000,)
    assert np.max(np.abs(y - 0.3)) < 1e-4


def test_wrong_sample_rate_is_rejected_with_a_clear_message(tmp_path: Path) -> None:
    path = tmp_path / "x.wav"
    sf.write(str(path), np.zeros(480), 48_000, subtype="PCM_16")
    with pytest.raises(ValueError, match="48000 Hz but 16000 Hz is required"):
        read_audio(path)


def test_resampling_keeps_frequency_and_amplitude(tmp_path: Path) -> None:
    n = np.arange(48_000)
    tone = 0.5 * np.sin(2.0 * np.pi * 1000.0 * n / 48_000)
    path = tmp_path / "t.wav"
    sf.write(str(path), tone, 48_000, subtype="PCM_24")
    y, fs = read_audio(path, allow_resample=True)
    assert fs == 16_000
    assert y.size == 16_000
    inner = y[2000:-2000]
    assert np.sqrt(np.mean(inner**2)) * np.sqrt(2.0) == pytest.approx(0.5, abs=0.01)
    spectrum = np.abs(rfft(y[:8192]))
    assert int(np.argmax(spectrum)) * 16_000 / 8192 == pytest.approx(1000.0, abs=4.0)


def test_resample_same_rate_is_identity(rng: np.random.Generator) -> None:
    x = rng.standard_normal(100)
    np.testing.assert_array_equal(resample(x, 16_000, 16_000), x)


def test_resample_rejects_bad_rates() -> None:
    with pytest.raises(ValueError):
        resample(np.zeros(10), 0, 16_000)


def test_float_to_pcm16_clips_and_counts() -> None:
    samples, clipped = float_to_pcm16([0.0, 0.5, 1.0, 2.0, -1.0, -3.0])
    assert samples.tolist() == [0, 16384, 32767, 32767, -32768, -32768]
    assert clipped == 3
    assert samples.dtype == np.int16


def test_write_wav_reports_clipping(tmp_path: Path) -> None:
    assert write_wav(tmp_path / "c.wav", [0.0, 1.5, -1.5, 0.2]) == 2


def test_write_wav_accepts_stereo_and_makes_mono(tmp_path: Path) -> None:
    write_wav(tmp_path / "m.wav", np.zeros((100, 2)))
    assert sf.info(str(tmp_path / "m.wav")).channels == 1


def test_limit_peak_scales_down_but_never_up() -> None:
    quiet, gain_quiet = limit_peak([0.1, -0.2])
    assert gain_quiet == 1.0
    np.testing.assert_array_equal(quiet, [0.1, -0.2])
    loud, gain_loud = limit_peak([0.5, -2.0], max_peak=0.9)
    assert np.max(np.abs(loud)) == pytest.approx(0.9)
    assert gain_loud == pytest.approx(0.45)
    with pytest.raises(ValueError):
        limit_peak([1.0], max_peak=0.0)
    assert limit_peak([], 0.9)[1] == 1.0


def test_invalid_audio_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        to_mono([])
    with pytest.raises(ValueError):
        to_mono(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        float_to_pcm16([0.0, float("nan")])
    assert to_mono(np.ones(5)).shape == (5,)
