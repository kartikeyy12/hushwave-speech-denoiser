"""Baselines B0 to B4 and the shared offline spectral pipeline (docs/THEORY.md section 7.6).

B0  unprocessed input (the reference every metric improvement is measured against)
B1  our magnitude spectral subtraction (G1), no temporal gain smoothing
B2  scipy.signal.wiener: a time-domain local-statistics Wiener filter, untuned off-the-shelf
B3  noisereduce (non-stationary spectral gating), an independent third-party implementation
B4  our Wiener gain with the instantaneous a-priori SNR (no decision-directed smoothing)

B1 and B4 use the same STFT, the same noise tracker and the same synthesis as the full system, so
that differences between B1, B4 and the full system (B5, M6) come from the gain rule alone.
B5 is built in M6 on the streaming engine; ``spectral_denoise`` is its offline reference.
"""

from __future__ import annotations

import importlib

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import wiener

from hushwave.config import DEFAULT, Config
from hushwave.gains import GainStage
from hushwave.noise_est import make_estimator
from hushwave.stft import istft, stft

FloatArray = NDArray[np.float64]
BASELINE_NAMES = ("B0", "B1", "B2", "B3", "B4")
B2_WINDOW = 15


def _as_signal(x: ArrayLike) -> FloatArray:
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"expected a mono 1-D signal, got shape {signal.shape}")
    if signal.size == 0:
        raise ValueError("signal is empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or Inf")
    return signal


def spectral_denoise(
    x: ArrayLike,
    rule: str,
    estimator: str = "mcra",
    cfg: Config = DEFAULT,
    gain_overrides: dict[str, float] | None = None,
    estimator_overrides: dict[str, float] | None = None,
) -> FloatArray:
    """STFT, noise PSD tracking, gain rule, inverse STFT. Returns exactly len(x) samples."""
    signal = _as_signal(x)
    spectrum = stft(signal, cfg.frame_len, cfg.hop)
    power = np.abs(spectrum) ** 2
    tracker = make_estimator(estimator, cfg.n_bins, cfg, **(estimator_overrides or {}))
    stage = GainStage(rule, cfg.n_bins, cfg, **(gain_overrides or {}))
    gains = np.empty_like(power)
    for index, frame in enumerate(power):
        gains[index] = stage.process(frame, tracker.update(frame))
    return istft(spectrum * gains, signal.size, cfg.frame_len, cfg.hop)


def b0_unprocessed(x: ArrayLike) -> FloatArray:
    """B0: the input, unchanged."""
    return _as_signal(x).copy()


def b1_spectral_subtraction(
    x: ArrayLike, cfg: Config = DEFAULT, estimator: str = "mcra"
) -> FloatArray:
    """B1: G1 on the shared pipeline with temporal smoothing switched off (classical form)."""
    return spectral_denoise(x, "subtraction", estimator, cfg, gain_overrides={"gain_mu": 0.0})


def b2_scipy_wiener(x: ArrayLike, window: int = B2_WINDOW) -> FloatArray:
    """B2: scipy.signal.wiener with an odd window of `window` samples and its own noise estimate.

    scipy divides by the local variance, so an all-zero input would produce NaN; silence is
    returned as silence.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer of at least 3")
    signal = _as_signal(x)
    if not np.any(signal):
        return np.zeros_like(signal)
    return np.asarray(wiener(signal, window), dtype=np.float64)


def b3_noisereduce(x: ArrayLike, fs: int = DEFAULT.fs) -> FloatArray:
    """B3: noisereduce.reduce_noise in non-stationary mode (needs `pip install noisereduce`)."""
    signal = _as_signal(x)
    try:
        module = importlib.import_module("noisereduce")
    except ImportError as error:
        raise ImportError(
            "baseline B3 needs the optional package noisereduce: pip install -e '.[metrics]'"
        ) from error
    if not np.any(signal):
        return np.zeros_like(signal)
    out = np.asarray(module.reduce_noise(y=signal, sr=fs, stationary=False), dtype=np.float64)
    if out.shape != signal.shape:
        raise RuntimeError(f"noisereduce returned shape {out.shape}, expected {signal.shape}")
    return out


def b4_plain_wiener(x: ArrayLike, cfg: Config = DEFAULT, estimator: str = "mcra") -> FloatArray:
    """B4: Wiener gain with xi = max(gamma - 1, 0) (no decision-directed smoothing)."""
    return spectral_denoise(x, "ml_wiener", estimator, cfg, gain_overrides={"gain_mu": 0.0})


def run_baseline(
    name: str, x: ArrayLike, cfg: Config = DEFAULT, estimator: str = "mcra"
) -> FloatArray:
    """Run one baseline by its label (B0 to B4)."""
    if name == "B0":
        return b0_unprocessed(x)
    if name == "B1":
        return b1_spectral_subtraction(x, cfg, estimator)
    if name == "B2":
        return b2_scipy_wiener(x)
    if name == "B3":
        return b3_noisereduce(x, cfg.fs)
    if name == "B4":
        return b4_plain_wiener(x, cfg, estimator)
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")
