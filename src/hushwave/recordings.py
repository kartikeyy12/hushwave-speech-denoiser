"""Checks for your own recordings (see docs/RECORDING_PROTOCOL.md)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from hushwave.audio_io import TARGET_FS, read_audio, write_wav

FLOOR_DB = -120.0


@dataclass(frozen=True)
class RecordingReport:
    duration_s: float
    rms_dbfs: float
    peak_dbfs: float
    clipped_fraction: float
    dc_offset: float
    quiet_frame_dbfs: float
    warnings: tuple[str, ...]


def _db(amplitude: float) -> float:
    return FLOOR_DB if amplitude <= 0.0 else max(FLOOR_DB, 20.0 * math.log10(amplitude))


def analyze_recording(
    x: ArrayLike, fs: int = TARGET_FS, min_duration_s: float = 30.0
) -> RecordingReport:
    """Level, clipping and gating checks on a mono float recording in [-1, 1]."""
    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("expected a non-empty 1-D recording")
    duration = data.size / fs
    rms = float(np.sqrt(np.mean(data**2)))
    peak = float(np.max(np.abs(data)))
    clipped = float(np.mean(np.abs(data) >= 0.999))
    dc = float(np.mean(data))
    frame = 512
    count = data.size // frame
    if count >= 1:
        frame_rms = np.sqrt(np.mean(data[: count * frame].reshape(count, frame) ** 2, axis=1))
        quiet = _db(float(np.percentile(frame_rms, 10)))
    else:
        quiet = _db(rms)
    warnings: list[str] = []
    if duration < min_duration_s:
        warnings.append(f"too short: {duration:.1f} s, want at least {min_duration_s:.0f} s")
    if clipped > 1e-3:
        warnings.append(f"clipping: {100 * clipped:.2f} % of samples at full scale; lower the gain")
    if _db(peak) > -1.0:
        warnings.append("peak is within 1 dB of full scale; lower the gain")
    if _db(peak) < -30.0:
        warnings.append("very quiet (peak below -30 dBFS); raise the gain or move closer")
    if abs(dc) > 0.01:
        warnings.append(f"large DC offset ({dc:.3f})")
    if quiet < -85.0:
        warnings.append("near-silent frames found; the recorder may use a noise gate or AGC")
    return RecordingReport(duration, _db(rms), _db(peak), clipped, dc, quiet, tuple(warnings))


def prepare_recording(
    source: Path, destination: Path, min_duration_s: float = 30.0
) -> RecordingReport:
    """Read any soundfile-readable file, resample to 16 kHz mono, report, and save as WAV."""
    samples, fs = read_audio(source, TARGET_FS, allow_resample=True)
    report = analyze_recording(samples, fs, min_duration_s)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_wav(destination, samples, fs)
    return report
