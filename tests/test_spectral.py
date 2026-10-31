"""Our Welch estimator against scipy.signal.welch and against known PSD levels."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch as scipy_welch

from hushwave.spectral import welch
from hushwave.windows import sqrt_hann

FS = 16_000.0


@pytest.mark.parametrize(
    ("nperseg", "noverlap", "scaling"),
    [
        (256, None, "density"),
        (256, 128, "spectrum"),
        (512, 256, "density"),
        (128, 0, "density"),
        (64, 48, "spectrum"),
    ],
)
def test_matches_scipy_welch(
    rng: np.random.Generator, nperseg: int, noverlap: int | None, scaling: str
) -> None:
    x = rng.standard_normal(20_000) + 0.7
    freqs, psd = welch(x, FS, nperseg=nperseg, noverlap=noverlap, scaling=scaling)
    freqs_ref, psd_ref = scipy_welch(
        x,
        FS,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling=scaling,
    )
    np.testing.assert_allclose(freqs, freqs_ref, atol=1e-9)
    np.testing.assert_allclose(psd, psd_ref, rtol=1e-9, atol=1e-18)


def test_matches_scipy_welch_with_custom_window(rng: np.random.Generator) -> None:
    x = rng.standard_normal(10_000)
    window = sqrt_hann(256)
    _, psd = welch(x, FS, window=window)
    _, psd_ref = scipy_welch(x, FS, window=window, nperseg=256, detrend="constant")
    np.testing.assert_allclose(psd, psd_ref, rtol=1e-9)


def test_white_noise_density_level(rng: np.random.Generator) -> None:
    sigma = 0.3
    x = sigma * rng.standard_normal(1 << 18)
    _, psd = welch(x, FS)
    expected = 2.0 * sigma**2 / FS
    assert float(np.mean(psd[1:-1])) == pytest.approx(expected, rel=0.03)


def test_integral_of_density_is_the_variance(rng: np.random.Generator) -> None:
    x = rng.standard_normal(1 << 17)
    freqs, psd = welch(x, FS)
    area = float(np.sum(psd) * (freqs[1] - freqs[0]))
    assert area == pytest.approx(float(np.var(x)), rel=0.03)


def test_sinusoid_peak_is_at_its_frequency() -> None:
    n = np.arange(1 << 14)
    x = np.sin(2.0 * np.pi * 1000.0 * n / FS)
    freqs, psd = welch(x, FS)
    assert freqs[int(np.argmax(psd))] == pytest.approx(1000.0, abs=FS / 256)


def test_zero_and_dc_inputs_give_zero_psd() -> None:
    assert not np.any(welch(np.zeros(2000), FS)[1])
    assert np.max(welch(np.full(2000, 5.0), FS)[1]) < 1e-20


def test_frequency_axis() -> None:
    freqs, psd = welch(np.zeros(1000), FS, nperseg=256)
    assert freqs.shape == psd.shape == (129,)
    assert freqs[0] == 0.0
    assert freqs[-1] == FS / 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nperseg": 100},
        {"nperseg": 1},
        {"nperseg": 4096},
        {"noverlap": 256},
        {"noverlap": -1},
        {"scaling": "power"},
        {"fs": 0.0},
        {"window": np.ones(10)},
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, object]) -> None:
    arguments: dict[str, object] = {"fs": FS, "nperseg": 256}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        welch(np.zeros(1000), **arguments)  # type: ignore[arg-type]


def test_invalid_signals_are_rejected() -> None:
    with pytest.raises(ValueError):
        welch(np.zeros((2, 500)), FS)
    with pytest.raises(ValueError):
        welch(np.array([1.0] * 300 + [np.inf]), FS)
