"""Closed-form results and the aliasing demonstration used as D1 evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin


def quantization_sqnr_db(bits: int, level_db_below_full_scale: float = 0.0) -> float:
    """SQNR of a uniform N-bit quantizer for a sine that is `level_db_below_full_scale` dB
    below a full-scale sine: 20 N log10(2) + 10 log10(1.5) - level  (about 6.02 N + 1.76)."""
    if bits < 1:
        raise ValueError("bits must be at least 1")
    if level_db_below_full_scale < 0.0:
        raise ValueError("level_db_below_full_scale must be non-negative")
    return 20.0 * bits * math.log10(2.0) + 10.0 * math.log10(1.5) - level_db_below_full_scale


def aliased_frequency_hz(freq_hz: float, fs: float) -> float:
    """Frequency in [0, fs/2] at which a tone of `freq_hz` appears after sampling at `fs`."""
    if freq_hz < 0.0 or fs <= 0.0:
        raise ValueError("need freq_hz >= 0 and fs > 0")
    folded = math.fmod(freq_hz, fs)
    return fs - folded if folded > fs / 2 else folded


def butterworth_highpass_gain_db(freq_hz: float, cutoff_hz: float, fs: float) -> float:
    """Magnitude (dB) of the 2nd-order Butterworth high-pass designed with the bilinear
    transform: |H|^2 = 1 / (1 + (tan(pi fc / fs) / tan(pi f / fs))^4)."""
    if not 0.0 < freq_hz < fs / 2 or not 0.0 < cutoff_hz < fs / 2:
        raise ValueError("frequencies must lie strictly between 0 and Nyquist")
    ratio = math.tan(math.pi * cutoff_hz / fs) / math.tan(math.pi * freq_hz / fs)
    return -10.0 * math.log10(1.0 + ratio**4)


@dataclass(frozen=True)
class AliasingResult:
    """Outcome of decimating a two-tone signal with and without an anti-alias filter."""

    source_fs: int
    target_fs: int
    factor: int
    numtaps: int
    cutoff_hz: float
    in_band_hz: float
    out_of_band_hz: float
    predicted_alias_hz: float
    naive_alias_db: float
    filtered_alias_db: float
    freqs_hz: NDArray[np.float64]
    naive_spectrum_db: NDArray[np.float64]
    filtered_spectrum_db: NDArray[np.float64]


def _spectrum_db(x: NDArray[np.float64]) -> NDArray[np.float64]:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.asarray(20.0 * np.log10(np.maximum(spectrum, 1e-300)), dtype=np.float64)


def _peak_near(
    spectrum_db: NDArray[np.float64], freq_hz: float, bin_hz: float, half_width_bins: int = 3
) -> float:
    centre = round(freq_hz / bin_hz)
    low = max(centre - half_width_bins, 0)
    return float(np.max(spectrum_db[low : centre + half_width_bins + 1]))


def aliasing_experiment(
    source_fs: int = 48_000,
    factor: int = 3,
    in_band_hz: float = 1_000.0,
    out_of_band_hz: float = 11_000.0,
    duration_s: float = 0.5,
    numtaps: int = 255,
    cutoff_hz: float = 7_200.0,
) -> AliasingResult:
    """Decimate sin(2 pi f1 t) + sin(2 pi f2 t) by `factor`, with and without a low-pass.

    `f2` lies above the target Nyquist, so plain sample dropping folds it onto
    aliased_frequency_hz(f2, target_fs). The filtered path applies a symmetric linear-phase
    FIR (scipy.signal.firwin, a reference design used only for this demonstration) first.
    Levels are in dB relative to the in-band tone's peak in the same spectrum.
    """
    if source_fs % factor != 0:
        raise ValueError("source_fs must be divisible by factor")
    target_fs = source_fs // factor
    t = np.arange(round(duration_s * source_fs), dtype=np.float64) / source_fs
    signal = np.sin(2.0 * np.pi * in_band_hz * t) + np.sin(2.0 * np.pi * out_of_band_hz * t)

    naive = signal[::factor]
    taps = firwin(numtaps, cutoff_hz, fs=source_fs)
    filtered = np.convolve(signal, taps, mode="same")[::factor]

    naive_db = _spectrum_db(naive)
    filtered_db = _spectrum_db(filtered)
    bin_hz = target_fs / len(naive)
    alias_hz = aliased_frequency_hz(out_of_band_hz, target_fs)

    naive_alias = _peak_near(naive_db, alias_hz, bin_hz) - _peak_near(naive_db, in_band_hz, bin_hz)
    filtered_alias = _peak_near(filtered_db, alias_hz, bin_hz) - _peak_near(
        filtered_db, in_band_hz, bin_hz
    )
    freqs = np.fft.rfftfreq(len(naive), d=1.0 / target_fs)
    reference_naive = _peak_near(naive_db, in_band_hz, bin_hz)
    reference_filtered = _peak_near(filtered_db, in_band_hz, bin_hz)
    return AliasingResult(
        source_fs=source_fs,
        target_fs=target_fs,
        factor=factor,
        numtaps=numtaps,
        cutoff_hz=cutoff_hz,
        in_band_hz=in_band_hz,
        out_of_band_hz=out_of_band_hz,
        predicted_alias_hz=alias_hz,
        naive_alias_db=naive_alias,
        filtered_alias_db=filtered_alias,
        freqs_hz=np.asarray(freqs, dtype=np.float64),
        naive_spectrum_db=naive_db - reference_naive,
        filtered_spectrum_db=filtered_db - reference_filtered,
    )
