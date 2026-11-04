"""Spectral gain rules and the stateful gain stage (docs/THEORY.md section 7).

Notation, per time-frequency cell: Y is the noisy STFT coefficient, lambda_d the noise PSD
estimate, gamma = |Y|^2 / lambda_d the a-posteriori SNR, xi the a-priori SNR. Every rule returns a
gain G with g_min <= G <= 1; the enhanced coefficient is G * Y.

Rules
-----
``subtraction``  G1, magnitude spectral subtraction with over-subtraction a and floor g_min.
``wiener``       G2 = xi / (1 + xi) with the decision-directed xi.
``lsa``          G3, MMSE log-spectral amplitude with the decision-directed xi, clipped to 1.
``ml_wiener``    B4 baseline: G2 with the instantaneous estimate xi = max(gamma - 1, 0).

Design decisions made here (and recorded in THEORY.md):

* The decision-directed recursion feeds back the amplitude produced by the rule after the gain
  floor but before temporal smoothing, so the smoother is a pure output stage and does not change
  how xi evolves.
* The recursion starts from a previous SNR of zero. The first frames are the least informative
  ones (half the window energy, a noise estimate built from one frame), so the safe start is
  strong attenuation, never a burst of unattenuated noise.
* There is no separate floor on xi: the gain floor g_min already bounds the output, and a floor
  on xi would be a second parameter doing the same job.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import exp1

from hushwave.config import DEFAULT, Config

FloatArray = NDArray[np.float64]
RULES = ("subtraction", "wiener", "lsa", "ml_wiener")


def _as_float_array(value: ArrayLike, name: str) -> FloatArray:
    data = np.asarray(value, dtype=np.float64)
    if np.any(np.isnan(data)):
        raise ValueError(f"{name} contains NaN")
    return data


def spectral_highpass_mask(n_bins: int, cutoff_hz: float, fs: float) -> FloatArray:
    """Zero-phase high-pass as a real per-bin gain (THEORY 7.7).

    The magnitude is that of the project's 2nd-order Butterworth biquad (bilinear transform),
    |H(f)| = 1 / sqrt(1 + (tan(pi fc / fs) / tan(pi f / fs))^4), sampled at the bin centres
    f_k = k fs / N with N = 2 (n_bins - 1). The DC bin gets 0 and the Nyquist bin 1. A real,
    positive gain has no phase, so unlike the causal biquad it does not shift the low harmonics
    of the voice against the unfiltered clean reference.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if not 0.0 < cutoff_hz < fs / 2:
        raise ValueError("cutoff_hz must lie strictly between 0 and Nyquist")
    freqs = np.arange(n_bins, dtype=np.float64) * fs / (2 * (n_bins - 1))
    mask = np.ones(n_bins)
    mask[0] = 0.0
    inner = slice(1, n_bins - 1)
    ratio = np.tan(np.pi * cutoff_hz / fs) / np.tan(np.pi * freqs[inner] / fs)
    mask[inner] = 1.0 / np.sqrt(1.0 + ratio**4)
    return mask


def subtraction_gain(gamma: ArrayLike, a: float, g_min: float) -> FloatArray:
    """G1 = max(1 - a / sqrt(gamma), g_min); gamma = 0 maps to the floor."""
    snr = _as_float_array(gamma, "gamma")
    if np.any(snr < 0.0):
        raise ValueError("gamma must be non-negative")
    if a < 1.0:
        raise ValueError("a must be at least 1")
    if not 0.0 < g_min <= 1.0:
        raise ValueError("g_min must lie in (0, 1]")
    root = np.sqrt(snr)
    ratio = np.full_like(root, np.inf)
    np.divide(a, root, out=ratio, where=root > 0.0)
    return np.asarray(np.clip(1.0 - ratio, g_min, 1.0), dtype=np.float64)


def _wiener_raw(xi: FloatArray) -> FloatArray:
    finite = np.where(np.isinf(xi), 0.0, xi)
    return np.asarray(np.where(np.isinf(xi), 1.0, finite / (1.0 + finite)), dtype=np.float64)


def _check_xi(xi: ArrayLike) -> FloatArray:
    data = _as_float_array(xi, "xi")
    if np.any(data < 0.0):
        raise ValueError("xi must be non-negative")
    return data


def wiener_gain(xi: ArrayLike, g_min: float) -> FloatArray:
    """G2 = xi / (1 + xi), clamped to [g_min, 1]."""
    if not 0.0 < g_min <= 1.0:
        raise ValueError("g_min must lie in (0, 1]")
    return np.asarray(np.clip(_wiener_raw(_check_xi(xi)), g_min, 1.0), dtype=np.float64)


def lsa_gain_raw(xi: ArrayLike, gamma: ArrayLike) -> FloatArray:
    """Unclipped MMSE-LSA gain G2 * exp(0.5 * E1(v)), v = G2 * gamma (can exceed 1).

    xi = 0 gives 0 (nothing to estimate); gamma = 0 with xi > 0 gives +inf (the limit of the
    gain as |Y| tends to 0), which the caller clips.
    """
    prior = _check_xi(xi)
    post = _as_float_array(gamma, "gamma")
    if np.any(post < 0.0):
        raise ValueError("gamma must be non-negative")
    g2 = _wiener_raw(prior)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw = g2 * np.exp(0.5 * exp1(g2 * post))
    return np.asarray(np.where(g2 > 0.0, raw, 0.0), dtype=np.float64)


def lsa_gain(xi: ArrayLike, gamma: ArrayLike, g_min: float) -> FloatArray:
    """G3 = clamp(G3_raw, g_min, 1): the clip at 1 is a project design decision (THEORY 7.3)."""
    if not 0.0 < g_min <= 1.0:
        raise ValueError("g_min must lie in (0, 1]")
    return np.asarray(np.clip(lsa_gain_raw(xi, gamma), g_min, 1.0), dtype=np.float64)


def noise_only_floor_escape(a: float, g_min: float) -> float:
    """P(G1 > g_min) for a noise-only bin when the noise PSD estimate is exact.

    In a noise-only bin gamma ~ Exp(1). G1 leaves the floor when 1 - a / sqrt(gamma) > g_min,
    i.e. gamma > (a / (1 - g_min))^2, which has probability exp(-(a / (1 - g_min))^2).
    """
    if a < 1.0:
        raise ValueError("a must be at least 1")
    if not 0.0 <= g_min < 1.0:
        raise ValueError("g_min must lie in [0, 1)")
    return math.exp(-((a / (1.0 - g_min)) ** 2))


class GainStage:
    """Per-frame gain computation with decision-directed xi and temporal gain smoothing."""

    def __init__(self, rule: str, n_bins: int, cfg: Config = DEFAULT, **overrides: float) -> None:
        if rule not in RULES:
            raise ValueError(f"unknown gain rule {rule!r}; choose from {RULES}")
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        unknown = set(overrides) - {"ss_a", "g_min", "gain_mu", "dd_alpha", "spectral_hp"}
        if unknown:
            raise ValueError(f"unknown gain parameters: {', '.join(sorted(unknown))}")
        params = {
            "ss_a": cfg.ss_a,
            "g_min": cfg.g_min,
            "gain_mu": cfg.gain_mu,
            "dd_alpha": cfg.dd_alpha,
            "spectral_hp": 1.0 if cfg.spectral_hp else 0.0,
        }
        params.update(overrides)
        if params["ss_a"] < 1.0:
            raise ValueError("ss_a must be at least 1")
        if not 0.0 < params["g_min"] <= 1.0:
            raise ValueError("g_min must lie in (0, 1]")
        if not 0.0 <= params["gain_mu"] < 1.0:
            raise ValueError("gain_mu must lie in [0, 1)")
        if not 0.0 < params["dd_alpha"] < 1.0:
            raise ValueError("dd_alpha must lie strictly between 0 and 1")
        self.rule = rule
        self.n_bins = n_bins
        self.a = float(params["ss_a"])
        self.g_min = float(params["g_min"])
        self.mu = float(params["gain_mu"])
        self.alpha = float(params["dd_alpha"])
        self._hp_mask: FloatArray | None = None
        if params["spectral_hp"] >= 0.5:
            self._hp_mask = spectral_highpass_mask(n_bins, cfg.hp_cutoff_hz, cfg.fs)
        self.reset()

    def reset(self) -> None:
        self._prev_snr = np.zeros(self.n_bins)
        self._smoothed: FloatArray | None = None
        self.xi: FloatArray = np.zeros(self.n_bins)

    def process(self, power: ArrayLike, noise_psd: ArrayLike) -> FloatArray:
        """Gain for one frame from its power |Y|^2 and the noise PSD estimate lambda_d."""
        frame = np.asarray(power, dtype=np.float64)
        noise = np.asarray(noise_psd, dtype=np.float64)
        for name, data in (("power", frame), ("noise_psd", noise)):
            if data.shape != (self.n_bins,):
                raise ValueError(f"{name} must have shape ({self.n_bins},), got {data.shape}")
            if not np.all(np.isfinite(data)):
                raise ValueError(f"{name} must be finite")
        if np.any(frame < 0.0):
            raise ValueError("power must be non-negative")
        if np.any(noise <= 0.0):
            raise ValueError("noise_psd must be strictly positive")
        gamma = frame / noise
        ml_xi = np.maximum(gamma - 1.0, 0.0)
        if self.rule == "subtraction":
            raw = subtraction_gain(gamma, self.a, self.g_min)
        elif self.rule == "ml_wiener":
            self.xi = ml_xi
            raw = wiener_gain(ml_xi, self.g_min)
        else:
            self.xi = self.alpha * self._prev_snr + (1.0 - self.alpha) * ml_xi
            if self.rule == "wiener":
                raw = wiener_gain(self.xi, self.g_min)
            else:
                raw = lsa_gain(self.xi, gamma, self.g_min)
            self._prev_snr = raw * raw * gamma
        if self._smoothed is None or self.mu == 0.0:
            self._smoothed = raw
        else:
            self._smoothed = self.mu * self._smoothed + (1.0 - self.mu) * raw
        if self._hp_mask is not None:
            return np.asarray(self._smoothed * self._hp_mask, dtype=np.float64)
        return np.array(self._smoothed, dtype=np.float64)


def compute_gains(
    power_frames: ArrayLike,
    noise_frames: ArrayLike,
    rule: str,
    cfg: Config = DEFAULT,
    **overrides: float,
) -> FloatArray:
    """Run a GainStage over (frames, bins) arrays of |Y|^2 and lambda_d; one gain row per frame."""
    power = np.asarray(power_frames, dtype=np.float64)
    noise = np.asarray(noise_frames, dtype=np.float64)
    if power.ndim != 2 or power.shape != noise.shape:
        raise ValueError(
            f"expected matching (frames, bins) arrays, got {power.shape}, {noise.shape}"
        )
    if power.shape[0] == 0:
        raise ValueError("no frames to process")
    stage = GainStage(rule, power.shape[1], cfg, **overrides)
    return np.stack([stage.process(p, n) for p, n in zip(power, noise, strict=True)])
