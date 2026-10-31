"""Noise generators and exact-SNR mixing."""

from __future__ import annotations

import numpy as np
import pytest

from hushwave.spectral import welch
from hushwave.synth import (
    NOISE_KINDS,
    am_noise,
    babble_noise,
    hum_noise,
    make_noise,
    measured_snr_db,
    mix_at_snr,
    noise_segment,
    pink_noise,
    transient_noise,
    white_noise,
)

FS = 16_000.0
N = 1 << 18


def _slope(x: np.ndarray) -> float:
    freqs, psd = welch(x, FS, nperseg=1024)
    band = (freqs >= 100.0) & (freqs <= 4000.0)
    return float(np.polyfit(np.log10(freqs[band]), np.log10(psd[band]), 1)[0])


def test_white_noise_is_flat_and_unit_rms(rng: np.random.Generator) -> None:
    x = white_noise(N, rng)
    assert np.sqrt(np.mean(x**2)) == pytest.approx(1.0)
    assert _slope(x) == pytest.approx(0.0, abs=0.05)


def test_pink_noise_falls_ten_db_per_decade(rng: np.random.Generator) -> None:
    x = pink_noise(N, rng)
    assert np.sqrt(np.mean(x**2)) == pytest.approx(1.0)
    assert _slope(x) == pytest.approx(-1.0, abs=0.05)


def test_pink_noise_handles_lengths_that_are_not_powers_of_two(rng: np.random.Generator) -> None:
    assert pink_noise(1000, rng).shape == (1000,)


def test_hum_has_peaks_at_mains_harmonics(rng: np.random.Generator) -> None:
    x = hum_noise(N, FS, rng)
    freqs, psd = welch(x, FS, nperseg=8192)
    background = float(np.median(psd))
    for harmonic in (50.0, 100.0, 150.0):
        index = int(np.argmin(np.abs(freqs - harmonic)))
        assert 10.0 * np.log10(psd[index] / background) > 30.0


def test_am_noise_envelope_modulates_at_the_requested_rate(rng: np.random.Generator) -> None:
    x = am_noise(1 << 17, FS, rng, mod_hz=4.0, depth=0.8)
    spectrum = np.abs(np.fft.rfft(x**2 - np.mean(x**2)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / FS)
    band = (freqs > 0.5) & (freqs < 20.0)
    assert freqs[band][int(np.argmax(spectrum[band]))] == pytest.approx(4.0, abs=0.3)


def test_transient_noise_is_impulsive(rng: np.random.Generator) -> None:
    x = transient_noise(1 << 17, FS, rng)
    kurtosis = float(np.mean(x**4) / np.mean(x**2) ** 2)
    assert kurtosis > 10.0
    assert np.sqrt(np.mean(x**2)) == pytest.approx(1.0)


def test_babble_is_less_peaky_than_one_talker(rng: np.random.Generator) -> None:
    sources = [rng.standard_normal(16_000) * (rng.random(16_000) < 0.2) for _ in range(6)]
    single = sources[0] / np.sqrt(np.mean(sources[0] ** 2))
    babble = babble_noise(sources, 16_000, rng)
    assert np.mean(babble**4) < np.mean(single**4)
    assert np.sqrt(np.mean(babble**2)) == pytest.approx(1.0)


def test_generators_are_deterministic_and_seed_sensitive() -> None:
    first = pink_noise(2048, np.random.default_rng(5))
    again = pink_noise(2048, np.random.default_rng(5))
    other = pink_noise(2048, np.random.default_rng(6))
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_make_noise_dispatch(rng: np.random.Generator) -> None:
    sources = [rng.standard_normal(500), rng.standard_normal(500)]
    for kind in NOISE_KINDS:
        out = make_noise(kind, 4000, FS, rng, babble_sources=sources)
        assert out.shape == (4000,)
        assert np.all(np.isfinite(out))
    with pytest.raises(ValueError, match="unknown noise kind"):
        make_noise("rain", 100, FS, rng)
    with pytest.raises(ValueError, match="babble_sources"):
        make_noise("babble", 100, FS, rng)


def test_generator_arguments_are_validated(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError):
        white_noise(1, rng)
    with pytest.raises(ValueError):
        hum_noise(1000, 90.0, rng)
    with pytest.raises(ValueError):
        am_noise(1000, FS, rng, depth=1.0)
    with pytest.raises(ValueError):
        transient_noise(1000, FS, rng, rate_hz=0.0)
    with pytest.raises(ValueError):
        babble_noise([np.ones(10)], 100, rng)
    with pytest.raises(ValueError):
        babble_noise([np.ones(10), np.zeros(0)], 100, rng)
    with pytest.raises(ValueError, match="zero or non-finite power"):
        babble_noise([np.zeros(10), np.zeros(10)], 100, rng)


@pytest.mark.parametrize("snr", [-5.0, 0.0, 5.0, 10.0, 15.0])
def test_mix_hits_the_requested_snr_exactly(rng: np.random.Generator, snr: float) -> None:
    clean = rng.standard_normal(8000) * 0.3
    noise = rng.standard_normal(8000)
    mixture = mix_at_snr(clean, noise, snr)
    assert measured_snr_db(clean, mixture.noise) == pytest.approx(snr, abs=1e-9)
    np.testing.assert_allclose(mixture.noisy, clean + mixture.noise)


def test_mix_known_answers() -> None:
    ones = np.ones(100)
    assert mix_at_snr(ones, ones, 0.0).gain == pytest.approx(1.0)
    assert mix_at_snr(ones, ones, 20.0).gain == pytest.approx(0.1)
    assert mix_at_snr(ones, 2.0 * ones, 0.0).gain == pytest.approx(0.5)
    np.testing.assert_allclose(mix_at_snr(ones, ones, 20.0).noisy, 1.1 * ones)


def test_mix_rejects_undefined_cases() -> None:
    ones = np.ones(10)
    with pytest.raises(ValueError, match="zero power"):
        mix_at_snr(np.zeros(10), ones, 0.0)
    with pytest.raises(ValueError, match="zero power"):
        mix_at_snr(ones, np.zeros(10), 0.0)
    with pytest.raises(ValueError):
        mix_at_snr(ones, np.ones(11), 0.0)
    with pytest.raises(ValueError):
        mix_at_snr(ones, np.array([1.0] * 9 + [np.nan]), 0.0)
    with pytest.raises(ValueError):
        mix_at_snr(ones, ones, float("inf"))
    with pytest.raises(ValueError):
        measured_snr_db(ones, np.ones(3))


def test_noise_segment_lengths_and_determinism() -> None:
    noise = np.arange(1000, dtype=np.float64)
    long_piece = noise_segment(noise, 300, np.random.default_rng(1))
    assert long_piece.shape == (300,)
    assert np.all(np.diff(long_piece) == 1.0)
    np.testing.assert_array_equal(long_piece, noise_segment(noise, 300, np.random.default_rng(1)))
    tiled = noise_segment(noise[:100], 450, np.random.default_rng(2))
    assert tiled.shape == (450,)
    assert set(np.unique(tiled)) <= set(noise[:100])
    with pytest.raises(ValueError):
        noise_segment(np.zeros(0), 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        noise_segment(noise, 0, np.random.default_rng(0))
