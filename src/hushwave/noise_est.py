"""Noise power-spectral-density trackers: minimum statistics and MCRA (docs/THEORY.md section 6).

Both estimators are stateful and consume one power frame |Y(n, k)|^2 at a time, so the batch
helper and the streaming engine run exactly the same code. Neither assumes the recording starts
with silence.

Bias compensation. A minimum of a fluctuating positive quantity sits below its mean, so the raw
minimum-statistics output is biased low, and MCRA's gated update is biased slightly low as well.
Rather than copy constants from the literature, each estimator multiplies its output by a factor
B obtained by simulating that very estimator, with its own parameters, on stationary white
Gaussian noise passed through the project's own STFT, and setting B = 1 / mean(estimate / true PSD).

Start-up. The first STFT frame holds only half of the window energy (the signal is zero-padded in
front), and the window of the minimum fills over about 1.5 s. Smoothing therefore uses a
bias-corrected recursion (a weighted average of the frames seen so far), and the minimum-statistics
factor is a curve B[n] indexed by the number of frames seen, calibrated by many short simulations
through the same STFT, so the estimate is also unbiased while the window is still filling.
MCRA additionally leaves its minimum tracker idle for the first 2 / (1 - alpha_s) frames, so the
half-energy first frame cannot be mistaken for a speech-free floor.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hushwave.config import DEFAULT, Config
from hushwave.stft import stft
from hushwave.windows import sqrt_hann

FloatArray = NDArray[np.float64]
KINDS = ("minstats", "mcra")
CALIBRATION_FRAMES = 8000
CURVE_RUNS = 48
CALIBRATION_SEED = 20260102


class NoiseEstimator(Protocol):
    """What the gain stage needs from a noise tracker."""

    n_bins: int

    def update(self, power: ArrayLike) -> FloatArray: ...

    def reset(self) -> None: ...


class SlidingMin:
    """Minimum over the last 80 to 95 frames, kept as U sub-window minima of V frames each.

    The ring holds the minima of the last U - 1 completed sub-windows; the running minimum of the
    current sub-window is added on top. Just before a sub-window completes the span is U V - 1
    frames; right after it completes the span is (U - 1) V frames.
    """

    def __init__(self, n_bins: int, n_sub: int, sub_frames: int) -> None:
        if n_sub < 1 or sub_frames < 1:
            raise ValueError("n_sub and sub_frames must be at least 1")
        self.n_bins = n_bins
        self.n_sub = n_sub
        self.sub_frames = sub_frames
        self.reset()

    def reset(self) -> None:
        self._ring = np.full((self.n_sub - 1, self.n_bins), np.inf)
        self._write = 0
        self._current = np.full(self.n_bins, np.inf)
        self._count = 0

    def update(self, x: FloatArray) -> FloatArray:
        self._current = np.minimum(self._current, x)
        self._count += 1
        result = self._current.copy()
        if self.n_sub > 1:
            result = np.minimum(result, self._ring.min(axis=0))
        if self._count == self.sub_frames:
            if self.n_sub > 1:
                self._ring[self._write] = self._current
                self._write = (self._write + 1) % (self.n_sub - 1)
                result = np.minimum(self._ring.min(axis=0), np.inf)
            self._current = np.full(self.n_bins, np.inf)
            self._count = 0
        return np.asarray(result, dtype=np.float64)


def _check_power(power: ArrayLike, n_bins: int) -> FloatArray:
    data = np.asarray(power, dtype=np.float64)
    if data.shape != (n_bins,):
        raise ValueError(f"expected a power frame of shape ({n_bins},), got {data.shape}")
    if not np.all(np.isfinite(data)) or np.any(data < 0.0):
        raise ValueError("power frame must be finite and non-negative")
    return data


class MinStatsEstimator:
    """Simplified minimum statistics (Martin 2001): smooth, take the windowed minimum, de-bias."""

    def __init__(
        self,
        n_bins: int,
        alpha_s: float = DEFAULT.ms_alpha_s,
        n_sub: int = DEFAULT.ms_n_sub,
        sub_frames: int = DEFAULT.ms_sub_frames,
        bias: float | None = None,
        floor: float = DEFAULT.noise_floor,
    ) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if not 0.0 < alpha_s < 1.0:
            raise ValueError("alpha_s must lie strictly between 0 and 1")
        self.n_bins = n_bins
        self.alpha_s = alpha_s
        self.floor = floor
        self._min = SlidingMin(n_bins, n_sub, sub_frames)
        self._constant_bias = bias
        self._steady = 1.0
        self._curve: FloatArray = np.ones(0)
        if bias is None:
            keys = {"alpha_s": alpha_s, "n_sub": n_sub, "sub_frames": sub_frames}
            self._steady = calibrated_bias("minstats", n_bins, **keys)
            self._curve = np.asarray(calibrated_bias_curve(n_bins, **keys), dtype=np.float64)
        self.reset()

    @property
    def bias(self) -> float:
        """Bias factor currently in force (steady state once the window has filled)."""
        return self._steady if self._constant_bias is None else self._constant_bias

    def reset(self) -> None:
        self._min.reset()
        self._ema: FloatArray = np.zeros(self.n_bins)
        self._frames = 0

    def update(self, power: ArrayLike) -> FloatArray:
        frame = _check_power(power, self.n_bins)
        self._frames += 1
        self._ema = self.alpha_s * self._ema + (1.0 - self.alpha_s) * frame
        smooth = self._ema / (1.0 - self.alpha_s**self._frames)
        if self._constant_bias is not None:
            factor = self._constant_bias
        elif self._frames <= self._curve.size:
            factor = float(self._curve[self._frames - 1])
        else:
            factor = self._steady
        estimate = factor * self._min.update(smooth)
        return np.asarray(np.maximum(estimate, self.floor), dtype=np.float64)


class McraEstimator:
    """Minima-controlled recursive averaging (Cohen): fast update only where speech is absent."""

    def __init__(
        self,
        n_bins: int,
        alpha_s: float = DEFAULT.mcra_alpha_s,
        alpha_p: float = DEFAULT.mcra_alpha_p,
        alpha_d: float = DEFAULT.mcra_alpha_d,
        delta: float = DEFAULT.mcra_delta,
        n_sub: int = DEFAULT.ms_n_sub,
        sub_frames: int = DEFAULT.ms_sub_frames,
        bias: float | None = None,
        floor: float = DEFAULT.noise_floor,
    ) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        for name, value in (("alpha_s", alpha_s), ("alpha_p", alpha_p), ("alpha_d", alpha_d)):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        if delta <= 1.0:
            raise ValueError("delta must exceed 1")
        self.n_bins = n_bins
        self.alpha_s, self.alpha_p, self.alpha_d, self.delta = alpha_s, alpha_p, alpha_d, delta
        self.floor = floor
        self._min = SlidingMin(n_bins, n_sub, sub_frames)
        self._warmup = math.ceil(2.0 / (1.0 - alpha_s))
        self.bias = (
            calibrated_bias(
                "mcra",
                n_bins,
                alpha_s=alpha_s,
                alpha_p=alpha_p,
                alpha_d=alpha_d,
                delta=delta,
                n_sub=n_sub,
                sub_frames=sub_frames,
            )
            if bias is None
            else bias
        )
        self.reset()

    def reset(self) -> None:
        self._min.reset()
        self._ema = np.zeros(self.n_bins)
        self._frames = 0
        self._presence = np.zeros(self.n_bins)
        self._num = np.zeros(self.n_bins)
        self._den = np.zeros(self.n_bins)

    @staticmethod
    def _frequency_smooth(frame: FloatArray) -> FloatArray:
        padded = np.concatenate([frame[:1], frame, frame[-1:]])
        return np.asarray(
            0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:], dtype=np.float64
        )

    def update(self, power: ArrayLike) -> FloatArray:
        frame = _check_power(power, self.n_bins)
        self._frames += 1
        self._ema = self.alpha_s * self._ema + (1.0 - self.alpha_s) * self._frequency_smooth(frame)
        smooth = self._ema / (1.0 - self.alpha_s**self._frames)
        if self._frames <= self._warmup:
            indicator = np.zeros(self.n_bins)
        else:
            floor_of_smooth = self._min.update(smooth)
            indicator = (smooth > self.delta * floor_of_smooth).astype(np.float64)
        self._presence = self.alpha_p * self._presence + (1.0 - self.alpha_p) * indicator
        rate = self.alpha_d + (1.0 - self.alpha_d) * self._presence
        self._num = rate * self._num + (1.0 - rate) * frame
        self._den = rate * self._den + (1.0 - rate)
        return np.asarray(
            np.maximum(self.bias * self._num / self._den, self.floor), dtype=np.float64
        )


def make_estimator(
    kind: str, n_bins: int, cfg: Config = DEFAULT, **overrides: float
) -> MinStatsEstimator | McraEstimator:
    """Build an estimator of the given kind from the project configuration."""
    if kind == "minstats":
        params = {
            "alpha_s": cfg.ms_alpha_s,
            "n_sub": cfg.ms_n_sub,
            "sub_frames": cfg.ms_sub_frames,
            "floor": cfg.noise_floor,
        }
        params.update(overrides)
        return MinStatsEstimator(
            n_bins,
            alpha_s=float(params["alpha_s"]),
            n_sub=int(params["n_sub"]),
            sub_frames=int(params["sub_frames"]),
            bias=overrides.get("bias"),
            floor=float(params["floor"]),
        )
    if kind == "mcra":
        mparams = {
            "alpha_s": cfg.mcra_alpha_s,
            "alpha_p": cfg.mcra_alpha_p,
            "alpha_d": cfg.mcra_alpha_d,
            "delta": cfg.mcra_delta,
            "n_sub": cfg.ms_n_sub,
            "sub_frames": cfg.ms_sub_frames,
            "floor": cfg.noise_floor,
        }
        mparams.update(overrides)
        return McraEstimator(
            n_bins,
            alpha_s=float(mparams["alpha_s"]),
            alpha_p=float(mparams["alpha_p"]),
            alpha_d=float(mparams["alpha_d"]),
            delta=float(mparams["delta"]),
            n_sub=int(mparams["n_sub"]),
            sub_frames=int(mparams["sub_frames"]),
            bias=overrides.get("bias"),
            floor=float(mparams["floor"]),
        )
    raise ValueError(f"unknown estimator kind {kind!r}; choose from {KINDS}")


def estimate_noise_psd(
    power_frames: ArrayLike, kind: str = "mcra", cfg: Config = DEFAULT, **overrides: float
) -> FloatArray:
    """Run an estimator over a (frames, bins) power spectrogram; one estimate per frame."""
    power = np.asarray(power_frames, dtype=np.float64)
    if power.ndim != 2:
        raise ValueError(f"expected (frames, bins), got shape {power.shape}")
    estimator = make_estimator(kind, power.shape[1], cfg, **overrides)
    return np.stack([estimator.update(row) for row in power])


@cache
def calibrated_bias(kind: str, n_bins: int, **params: float) -> float:
    """B = 1 / mean(estimate / true PSD) on stationary white Gaussian noise (see module doc)."""
    frame_len = 2 * (n_bins - 1)
    hop = frame_len // 2
    rng = np.random.default_rng(CALIBRATION_SEED)
    noise = rng.standard_normal((CALIBRATION_FRAMES + 1) * hop)
    power = np.abs(stft(noise, frame_len, hop)) ** 2
    true_psd = float(np.sum(sqrt_hann(frame_len) ** 2))
    overrides = dict(params)
    overrides["bias"] = 1.0
    estimates = estimate_noise_psd(power, kind, DEFAULT, **overrides)
    window = int(
        params.get("n_sub", DEFAULT.ms_n_sub) * params.get("sub_frames", DEFAULT.ms_sub_frames)
    )
    steady = estimates[2 * window :, 1:-1] / true_psd
    return float(1.0 / np.mean(steady))


@cache
def calibrated_bias_curve(n_bins: int, **params: float) -> tuple[float, ...]:
    """Minimum-statistics factor B[n] for n = 1 .. 2 U V frames seen (start-up behaviour).

    Averaged over CURVE_RUNS independent noise realisations pushed through the project's own STFT,
    so it includes the half-energy first frame. Beyond the curve the steady-state factor is used.
    """
    frame_len = 2 * (n_bins - 1)
    hop = frame_len // 2
    length = 2 * int(
        params.get("n_sub", DEFAULT.ms_n_sub) * params.get("sub_frames", DEFAULT.ms_sub_frames)
    )
    true_psd = float(np.sum(sqrt_hann(frame_len) ** 2))
    rng = np.random.default_rng(CALIBRATION_SEED + 1)
    overrides = dict(params)
    overrides["bias"] = 1.0
    ratios = np.zeros((CURVE_RUNS, length))
    for run in range(CURVE_RUNS):
        noise = rng.standard_normal((length + 1) * hop)
        power = np.abs(stft(noise, frame_len, hop)) ** 2
        estimates = estimate_noise_psd(power, "minstats", DEFAULT, **overrides)
        ratios[run] = np.mean(estimates[:length, 1:-1], axis=1) / true_psd
    return tuple(float(v) for v in 1.0 / ratios.mean(axis=0))


@dataclass(frozen=True)
class StepResponse:
    """Frames needed to move halfway (in dB) to a new stationary noise level."""

    kind: str
    ratio_db: float
    delay_frames: int


def measure_step_delay(
    kind: str, ratio_db: float, n_bins: int = 257, seed: int = 7, settle: int = 400
) -> StepResponse:
    """Noise level jumps by `ratio_db` dB at frame `settle`; frames to the half-way point."""
    rng = np.random.default_rng(seed)
    amplitude = 10.0 ** (ratio_db / 20.0)
    before = rng.exponential(1.0, (settle, n_bins))
    after = amplitude**2 * rng.exponential(1.0, (settle + 800, n_bins))
    estimator = make_estimator(kind, n_bins)
    level: Callable[[FloatArray], float] = lambda est: float(10.0 * np.log10(np.mean(est)))  # noqa: E731
    for row in before:
        estimator.update(row)
    start = level(estimator.update(before[-1]))
    target = start + ratio_db / 2.0
    for index, row in enumerate(after):
        value = level(estimator.update(row))
        reached = value >= target if ratio_db > 0 else value <= target
        if reached:
            return StepResponse(kind, ratio_db, index + 1)
    return StepResponse(kind, ratio_db, len(after))
