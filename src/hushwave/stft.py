"""Short-time Fourier transform with weighted overlap-add (WOLA) reconstruction.

Layout. With frame length N and hop H = N / 2, the signal is zero-padded with P = N - H samples
in front (so the first real sample is covered by two frames) and enough zeros at the back that the
last real sample is also covered by two frames. Frame f spans padded samples [f H, f H + N).

Analysis multiplies each frame by the window w before the FFT; synthesis inverse-transforms,
multiplies by w again, and overlap-adds. Reconstruction is exact when w^2[n] + w^2[n + H] = 1,
which holds for the square root of the periodic Hann window (see docs/THEORY.md section 4.2).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from hushwave.fft import irfft, is_power_of_two, rfft
from hushwave.windows import sqrt_hann

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _check_framing(frame_len: int, hop: int) -> None:
    if not is_power_of_two(frame_len):
        raise ValueError(f"frame_len must be a power of two, got {frame_len}")
    if 2 * hop != frame_len:
        raise ValueError("hop must equal frame_len / 2 (50 % overlap)")


def _check_window(window: FloatArray, frame_len: int) -> None:
    if window.shape != (frame_len,):
        raise ValueError(f"window must have shape ({frame_len},), got {window.shape}")


def _as_signal(x: ArrayLike) -> FloatArray:
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"expected a mono 1-D signal, got shape {signal.shape}")
    if signal.size == 0:
        raise ValueError("signal is empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or Inf")
    return signal


def num_frames(n_samples: int, frame_len: int, hop: int) -> int:
    """Number of frames stft() produces for a signal of n_samples samples."""
    _check_framing(frame_len, hop)
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    lead = frame_len - hop
    return -(-(lead + n_samples) // hop)


def stft(
    x: ArrayLike, frame_len: int = 512, hop: int = 256, window: FloatArray | None = None
) -> ComplexArray:
    """STFT of a mono signal: complex array of shape (frames, frame_len / 2 + 1)."""
    _check_framing(frame_len, hop)
    signal = _as_signal(x)
    taper = sqrt_hann(frame_len) if window is None else np.asarray(window, dtype=np.float64)
    _check_window(taper, frame_len)
    lead = frame_len - hop
    frames = num_frames(signal.size, frame_len, hop)
    padded = np.zeros((frames - 1) * hop + frame_len, dtype=np.float64)
    padded[lead : lead + signal.size] = signal
    segments = sliding_window_view(padded, frame_len)[::hop]
    return rfft(segments * taper)


def istft(
    spectrum: ArrayLike,
    length: int,
    frame_len: int = 512,
    hop: int = 256,
    window: FloatArray | None = None,
) -> FloatArray:
    """Inverse STFT by windowed overlap-add; returns exactly `length` samples."""
    _check_framing(frame_len, hop)
    spec = np.asarray(spectrum, dtype=np.complex128)
    if spec.ndim != 2 or spec.shape[1] != frame_len // 2 + 1:
        raise ValueError(f"expected shape (frames, {frame_len // 2 + 1}), got {spec.shape}")
    if length < 1:
        raise ValueError("length must be at least 1")
    frames = spec.shape[0]
    if frames != num_frames(length, frame_len, hop):
        raise ValueError(
            f"{frames} frames do not match length {length} "
            f"(expected {num_frames(length, frame_len, hop)})"
        )
    taper = sqrt_hann(frame_len) if window is None else np.asarray(window, dtype=np.float64)
    _check_window(taper, frame_len)
    segments = irfft(spec, n=frame_len) * taper
    blocks = np.zeros((frames + 1, hop), dtype=np.float64)
    blocks[:frames] += segments[:, :hop]
    blocks[1:] += segments[:, hop:]
    lead = frame_len - hop
    return np.ascontiguousarray(blocks.reshape(-1)[lead : lead + length])
