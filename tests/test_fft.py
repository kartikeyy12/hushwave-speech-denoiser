"""FFT against numpy and against its own mathematical properties."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from hushwave.fft import fft, ifft, irfft, is_power_of_two, rfft

LENGTHS = [1, 2, 4, 8, 16, 64, 256, 512, 1024, 4096]


def _complex_noise(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("n", LENGTHS)
def test_fft_matches_numpy(rng: np.random.Generator, n: int) -> None:
    x = _complex_noise(rng, n)
    assert np.max(np.abs(fft(x) - np.fft.fft(x))) < 1e-9


@pytest.mark.parametrize("n", LENGTHS)
def test_ifft_matches_numpy_and_inverts(rng: np.random.Generator, n: int) -> None:
    x = _complex_noise(rng, n)
    assert np.max(np.abs(ifft(x) - np.fft.ifft(x))) < 1e-9
    assert np.max(np.abs(ifft(fft(x)) - x)) < 1e-9


def test_fft_is_batched_over_leading_axes(rng: np.random.Generator) -> None:
    x = _complex_noise(rng, 5, 3, 64)
    assert np.max(np.abs(fft(x) - np.fft.fft(x, axis=-1))) < 1e-9


@pytest.mark.parametrize("n", [2, 8, 512])
def test_rfft_matches_numpy(rng: np.random.Generator, n: int) -> None:
    x = rng.standard_normal((4, n))
    assert np.max(np.abs(rfft(x) - np.fft.rfft(x))) < 1e-9


@pytest.mark.parametrize("n", [2, 8, 512])
def test_irfft_matches_numpy_even_for_non_hermitian_input(rng: np.random.Generator, n: int) -> None:
    spectrum = _complex_noise(rng, 3, n // 2 + 1)
    assert np.max(np.abs(irfft(spectrum, n) - np.fft.irfft(spectrum, n))) < 1e-9


def test_irfft_default_length_and_round_trip(rng: np.random.Generator) -> None:
    x = rng.standard_normal(128)
    assert np.max(np.abs(irfft(rfft(x)) - x)) < 1e-9


def test_impulse_gives_flat_spectrum() -> None:
    impulse = np.zeros(64)
    impulse[0] = 1.0
    np.testing.assert_allclose(fft(impulse), np.ones(64), atol=1e-12)


def test_shifted_impulse_gives_linear_phase() -> None:
    n, shift = 64, 5
    impulse = np.zeros(n)
    impulse[shift] = 1.0
    expected = np.exp(-2j * np.pi * shift * np.arange(n) / n)
    np.testing.assert_allclose(fft(impulse), expected, atol=1e-12)


def test_on_bin_sinusoid_peaks_in_the_right_bin() -> None:
    n, bin_index = 512, 37
    x = np.cos(2.0 * np.pi * bin_index * np.arange(n) / n)
    magnitude = np.abs(rfft(x))
    assert int(np.argmax(magnitude)) == bin_index
    assert magnitude[bin_index] == pytest.approx(n / 2)


def test_parseval(rng: np.random.Generator) -> None:
    x = _complex_noise(rng, 256)
    assert np.sum(np.abs(fft(x)) ** 2) / 256 == pytest.approx(np.sum(np.abs(x) ** 2))


def test_all_zero_input_gives_exact_zeros() -> None:
    assert not np.any(fft(np.zeros(512)))


def test_dc_input_concentrates_in_bin_zero() -> None:
    spectrum = fft(np.full(256, 3.0))
    assert spectrum[0] == pytest.approx(768.0)
    assert np.max(np.abs(spectrum[1:])) < 1e-9


@pytest.mark.parametrize("bad_length", [0, 3, 6, 100, 511])
def test_non_power_of_two_is_rejected(bad_length: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        fft(np.zeros(bad_length))
    with pytest.raises(ValueError, match="power of two"):
        ifft(np.zeros(bad_length))


def test_scalar_inputs_are_rejected() -> None:
    for transform in (fft, ifft, rfft, irfft):
        with pytest.raises(ValueError):
            transform(1.0)


def test_irfft_rejects_inconsistent_length() -> None:
    with pytest.raises(ValueError, match="bins"):
        irfft(np.zeros(10, dtype=np.complex128), 64)


def test_is_power_of_two() -> None:
    assert [n for n in range(0, 20) if is_power_of_two(n)] == [1, 2, 4, 8, 16]


_floats = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(0, 8).flatmap(
        lambda p: st.tuples(
            hnp.arrays(np.float64, 2**p, elements=_floats),
            hnp.arrays(np.float64, 2**p, elements=_floats),
        )
    )
)
def test_fft_round_trip_and_linearity_property(pair: tuple[np.ndarray, np.ndarray]) -> None:
    real, imag = pair
    x = real + 1j * imag
    scale = 1.0 + np.max(np.abs(x))
    assert np.max(np.abs(ifft(fft(x)) - x)) < 1e-9 * scale
    assert np.max(np.abs(fft(2.0 * x + 1.0) - (2.0 * fft(x) + fft(np.ones_like(x))))) < 1e-9 * scale
