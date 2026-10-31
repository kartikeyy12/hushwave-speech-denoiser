"""Analysis/synthesis windows and numerical window metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def hann_periodic(length: int) -> NDArray[np.float64]:
    """Periodic Hann window 0.5 - 0.5 cos(2 pi n / N), n = 0..N-1."""
    if length < 2:
        raise ValueError("window length must be at least 2")
    n = np.arange(length, dtype=np.float64)
    return np.asarray(0.5 - 0.5 * np.cos(2.0 * np.pi * n / length), dtype=np.float64)


def sqrt_hann(length: int) -> NDArray[np.float64]:
    """Square root of the periodic Hann window, which equals sin(pi n / N)."""
    if length < 2:
        raise ValueError("window length must be at least 2")
    n = np.arange(length, dtype=np.float64)
    return np.asarray(np.sin(np.pi * n / length), dtype=np.float64)


@dataclass(frozen=True)
class WindowMetrics:
    """Spectral-leakage figures of a window, measured numerically."""

    enbw_bins: float
    mainlobe_null_to_null_bins: float
    peak_sidelobe_db: float


def window_metrics(window: NDArray[np.float64], oversample: int = 64) -> WindowMetrics:
    """Measure ENBW, null-to-null main-lobe width and peak side-lobe level.

    Frequency is expressed in bins of the length-N DFT, so a width of 3.0 means 3 bins.
    """
    n = len(window)
    if n < 2 or oversample < 2:
        raise ValueError("need a window of at least 2 samples and oversample >= 2")
    enbw = float(n * np.sum(window**2) / np.sum(window) ** 2)
    spectrum = np.abs(np.fft.rfft(window, n=n * oversample))
    first_minimum = int(np.argmax(np.diff(spectrum) > 0))
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum / spectrum[0], 1e-300))
    return WindowMetrics(
        enbw_bins=enbw,
        mainlobe_null_to_null_bins=2.0 * first_minimum / oversample,
        peak_sidelobe_db=float(np.max(spectrum_db[first_minimum:])),
    )
