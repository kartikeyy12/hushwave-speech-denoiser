"""Welch power-spectral-density estimate, written from first principles.

Split the signal into overlapping segments, remove each segment's mean, window, transform,
and average the periodograms. With window w and sampling rate fs, the one-sided density is

    P[k] = c * mean_segments |FFT(w * segment)[k]|^2,  c = 1 / (fs sum(w^2)),

doubled for 0 < k < N/2 to fold in the negative frequencies. ("spectrum" scaling uses
c = 1 / (sum w)^2 instead, giving power per bin.)
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from hushwave.fft import is_power_of_two, rfft
from hushwave.windows import hann_periodic

FloatArray = NDArray[np.float64]


def welch(
    x: ArrayLike,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: FloatArray | None = None,
    scaling: str = "density",
) -> tuple[FloatArray, FloatArray]:
    """One-sided Welch PSD. Returns (frequencies in Hz, estimate)."""
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {signal.shape}")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or Inf")
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    if not is_power_of_two(nperseg) or nperseg < 2:
        raise ValueError(f"nperseg must be a power of two >= 2, got {nperseg}")
    if signal.size < nperseg:
        raise ValueError(f"signal ({signal.size} samples) is shorter than nperseg ({nperseg})")
    overlap = nperseg // 2 if noverlap is None else noverlap
    if not 0 <= overlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    if scaling not in ("density", "spectrum"):
        raise ValueError("scaling must be 'density' or 'spectrum'")
    taper = hann_periodic(nperseg) if window is None else np.asarray(window, dtype=np.float64)
    if taper.shape != (nperseg,):
        raise ValueError(f"window must have shape ({nperseg},), got {taper.shape}")

    step = nperseg - overlap
    count = (signal.size - overlap) // step
    segments = sliding_window_view(signal, nperseg)[::step][:count]
    segments = segments - segments.mean(axis=1, keepdims=True)
    periodograms = np.abs(rfft(segments * taper)) ** 2
    if scaling == "density":
        scale = 1.0 / (fs * float(np.sum(taper**2)))
    else:
        scale = 1.0 / float(np.sum(taper)) ** 2
    psd = periodograms.mean(axis=0) * scale
    psd[1:-1] *= 2.0
    freqs = np.arange(nperseg // 2 + 1, dtype=np.float64) * fs / nperseg
    return freqs, np.asarray(psd, dtype=np.float64)
