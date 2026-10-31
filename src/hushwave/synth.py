"""Seeded noise generators and exact-SNR mixing (definition in docs/THEORY.md section 8)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hushwave.fft import irfft, rfft

FloatArray = NDArray[np.float64]
NOISE_KINDS = ("white", "pink", "hum", "babble", "am", "transient")


def _unit_rms(x: FloatArray) -> FloatArray:
    rms = float(np.sqrt(np.mean(x**2)))
    if rms == 0.0 or not np.isfinite(rms):
        raise ValueError("signal has zero or non-finite power")
    return np.asarray(x / rms, dtype=np.float64)


def _check_length(n: int) -> None:
    if n < 2:
        raise ValueError("need at least 2 samples")


def white_noise(n: int, rng: np.random.Generator) -> FloatArray:
    """Gaussian white noise, scaled to unit RMS."""
    _check_length(n)
    return _unit_rms(rng.standard_normal(n))


def pink_noise(n: int, rng: np.random.Generator) -> FloatArray:
    """1/f noise: white noise whose power spectrum is shaped by 1/k with our own FFT."""
    _check_length(n)
    size = 1 << (n - 1).bit_length()
    spectrum = rfft(rng.standard_normal(size))
    shape = np.zeros(size // 2 + 1)
    shape[1:] = 1.0 / np.sqrt(np.arange(1, size // 2 + 1))
    return _unit_rms(irfft(spectrum * shape, size)[:n])


def hum_noise(
    n: int,
    fs: float,
    rng: np.random.Generator,
    mains_hz: float = 50.0,
    n_harmonics: int = 8,
    floor_db: float = -40.0,
) -> FloatArray:
    """Mains hum: sines at k * mains_hz (amplitude 1/k, random phase) plus a weak noise floor."""
    _check_length(n)
    if mains_hz <= 0.0 or fs <= 2.0 * mains_hz:
        raise ValueError("need 0 < mains_hz < fs / 2")
    t = np.arange(n) / fs
    hum = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        if k * mains_hz >= fs / 2:
            break
        hum += np.sin(2.0 * np.pi * k * mains_hz * t + rng.uniform(0.0, 2.0 * np.pi)) / k
    hum = _unit_rms(hum)
    return _unit_rms(hum + 10.0 ** (floor_db / 20.0) * rng.standard_normal(n))


def am_noise(
    n: int, fs: float, rng: np.random.Generator, mod_hz: float = 4.0, depth: float = 0.8
) -> FloatArray:
    """White noise with a sinusoidal amplitude envelope 1 + depth * sin(2 pi f_m t + phase)."""
    _check_length(n)
    if not 0.0 <= depth < 1.0 or mod_hz <= 0.0:
        raise ValueError("need 0 <= depth < 1 and mod_hz > 0")
    t = np.arange(n) / fs
    envelope = 1.0 + depth * np.sin(2.0 * np.pi * mod_hz * t + rng.uniform(0.0, 2.0 * np.pi))
    return _unit_rms(envelope * rng.standard_normal(n))


def transient_noise(
    n: int,
    fs: float,
    rng: np.random.Generator,
    rate_hz: float = 3.0,
    decay_ms: float = 20.0,
    floor_db: float = -50.0,
) -> FloatArray:
    """Random clicks: noise bursts with an exponential decay on a very quiet floor."""
    _check_length(n)
    if rate_hz <= 0.0 or decay_ms <= 0.0:
        raise ValueError("rate_hz and decay_ms must be positive")
    tau = decay_ms * 1e-3 * fs
    out = 10.0 ** (floor_db / 20.0) * rng.standard_normal(n)
    count = max(1, int(rng.poisson(rate_hz * n / fs)))
    for start in rng.integers(0, n, size=count):
        length = min(n - int(start), int(6 * tau) + 1)
        out[start : start + length] += rng.standard_normal(length) * np.exp(
            -np.arange(length) / tau
        )
    return _unit_rms(out)


def babble_noise(sources: Sequence[ArrayLike], n: int, rng: np.random.Generator) -> FloatArray:
    """Sum of unit-RMS, randomly shifted copies of the given speech signals."""
    _check_length(n)
    if len(sources) < 2:
        raise ValueError("babble needs at least 2 source signals")
    total = np.zeros(n)
    for source in sources:
        data = np.asarray(source, dtype=np.float64)
        if data.ndim != 1 or data.size == 0:
            raise ValueError("each babble source must be a non-empty 1-D signal")
        tiled = np.resize(data, n)
        total += _unit_rms(np.roll(tiled, int(rng.integers(0, n))))
    return _unit_rms(total)


def make_noise(
    kind: str,
    n: int,
    fs: float,
    rng: np.random.Generator,
    babble_sources: Sequence[ArrayLike] | None = None,
) -> FloatArray:
    """Generate synthetic noise of one of NOISE_KINDS."""
    if kind == "white":
        return white_noise(n, rng)
    if kind == "pink":
        return pink_noise(n, rng)
    if kind == "hum":
        return hum_noise(n, fs, rng)
    if kind == "am":
        return am_noise(n, fs, rng)
    if kind == "transient":
        return transient_noise(n, fs, rng)
    if kind == "babble":
        if babble_sources is None:
            raise ValueError("babble noise needs babble_sources")
        return babble_noise(babble_sources, n, rng)
    raise ValueError(f"unknown noise kind {kind!r}; choose from {NOISE_KINDS}")


def noise_segment(noise: ArrayLike, n: int, rng: np.random.Generator) -> FloatArray:
    """A length-n piece of a longer noise recording (random start), or a tiled shorter one."""
    data = np.asarray(noise, dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("noise must be a non-empty 1-D signal")
    if n < 1:
        raise ValueError("n must be at least 1")
    if data.size >= n:
        start = int(rng.integers(0, data.size - n + 1))
        return np.asarray(data[start : start + n], dtype=np.float64)
    return np.asarray(
        np.roll(np.resize(data, n), int(rng.integers(0, data.size))), dtype=np.float64
    )


def measured_snr_db(clean: ArrayLike, noise: ArrayLike) -> float:
    """10 log10(sum clean^2 / sum noise^2) over the whole signal."""
    x = np.asarray(clean, dtype=np.float64)
    d = np.asarray(noise, dtype=np.float64)
    if x.shape != d.shape:
        raise ValueError("clean and noise must have the same shape")
    return float(10.0 * np.log10(np.sum(x**2) / np.sum(d**2)))


@dataclass(frozen=True)
class Mixture:
    """A noisy signal together with the exact noise component that was added."""

    noisy: FloatArray
    noise: FloatArray
    gain: float


def mix_at_snr(clean: ArrayLike, noise: ArrayLike, snr_db: float) -> Mixture:
    """y = x + g d with g = sqrt( mean(x^2) / (mean(d^2) 10^(snr/10)) ), so SNR(x, g d) = snr."""
    x = np.asarray(clean, dtype=np.float64)
    d = np.asarray(noise, dtype=np.float64)
    if x.ndim != 1 or x.shape != d.shape:
        raise ValueError("clean and noise must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(d)) and np.isfinite(snr_db)):
        raise ValueError("inputs must be finite")
    power_x = float(np.mean(x**2))
    power_d = float(np.mean(d**2))
    if power_x == 0.0:
        raise ValueError("clean signal has zero power; SNR is undefined")
    if power_d == 0.0:
        raise ValueError("noise has zero power; it cannot be scaled to a target SNR")
    gain = float(np.sqrt(power_x / (power_d * 10.0 ** (snr_db / 10.0))))
    scaled = gain * d
    return Mixture(noisy=x + scaled, noise=scaled, gain=gain)
