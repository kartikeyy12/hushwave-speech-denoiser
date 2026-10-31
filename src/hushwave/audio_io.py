"""Audio file input/output with the safety rules the rest of the project relies on.

soundfile does the actual file decoding and encoding (allowed: I/O only). Everything that
decides what happens to the samples lives here: mono down-mix, sample-rate policy, and
explicit clipping when converting to 16-bit PCM. Writing floats straight through libsndfile
would wrap around on overflow instead of clipping, so the conversion is done by hand.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import ArrayLike, NDArray
from scipy.signal import resample_poly

FloatArray = NDArray[np.float64]
PCM16_MAX = 32767
PCM16_MIN = -32768
TARGET_FS = 16_000


def to_mono(x: ArrayLike) -> FloatArray:
    """1-D stays 1-D; (samples, channels) is averaged over channels."""
    data = np.asarray(x, dtype=np.float64)
    if data.size == 0:
        raise ValueError("audio is empty")
    if data.ndim == 1:
        return data
    if data.ndim == 2:
        return np.asarray(data.mean(axis=1), dtype=np.float64)
    raise ValueError(f"expected 1-D or 2-D audio, got {data.ndim}-D")


def resample(x: ArrayLike, fs_in: int, fs_out: int) -> FloatArray:
    """Polyphase resampling with a proper anti-alias filter (scipy reference implementation)."""
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("sample rates must be positive")
    data = np.asarray(x, dtype=np.float64)
    if fs_in == fs_out:
        return data
    divisor = math.gcd(fs_in, fs_out)
    return np.asarray(resample_poly(data, fs_out // divisor, fs_in // divisor), dtype=np.float64)


def read_audio(
    path: str | Path, target_fs: int = TARGET_FS, allow_resample: bool = False
) -> tuple[FloatArray, int]:
    """Read a file as mono float64 at `target_fs`.

    A different sample rate is an error unless allow_resample=True, in which case the audio is
    resampled with an anti-alias filter.
    """
    data, fs = sf.read(str(path), dtype="float64", always_2d=True)
    mono = to_mono(data)
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"{path}: audio contains NaN or Inf")
    if fs != target_fs:
        if not allow_resample:
            raise ValueError(
                f"{path}: sample rate is {fs} Hz but {target_fs} Hz is required; "
                f"resample it (allow_resample=True) or convert it first"
            )
        mono = resample(mono, int(fs), target_fs)
    return mono, target_fs


def float_to_pcm16(x: ArrayLike) -> tuple[NDArray[np.int16], int]:
    """Round to 16-bit PCM, clipping out-of-range samples. Returns (samples, clipped count)."""
    data = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("cannot convert NaN or Inf to PCM")
    scaled = np.round(data * 32768.0)
    clipped = int(np.count_nonzero((scaled > PCM16_MAX) | (scaled < PCM16_MIN)))
    return np.clip(scaled, PCM16_MIN, PCM16_MAX).astype(np.int16), clipped


def limit_peak(x: ArrayLike, max_peak: float = 0.99) -> tuple[FloatArray, float]:
    """Scale down (never up) so that max |x| <= max_peak. Returns (signal, gain applied)."""
    data = np.asarray(x, dtype=np.float64)
    if not 0.0 < max_peak <= 1.0:
        raise ValueError("max_peak must lie in (0, 1]")
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= max_peak:
        return data, 1.0
    gain = max_peak / peak
    return data * gain, gain


def write_wav(path: str | Path, x: ArrayLike, fs: int = TARGET_FS) -> int:
    """Write mono 16-bit PCM WAV. Returns how many samples had to be clipped."""
    samples, clipped = float_to_pcm16(to_mono(x))
    sf.write(str(path), samples, fs, subtype="PCM_16")
    return clipped
