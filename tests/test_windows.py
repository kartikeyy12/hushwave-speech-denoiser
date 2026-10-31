"""Windows and their leakage metrics, against known closed-form values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hushwave.windows import hann_periodic, sqrt_hann, window_metrics


def test_sqrt_hann_squared_is_hann() -> None:
    np.testing.assert_allclose(sqrt_hann(512) ** 2, hann_periodic(512), atol=1e-15)


def test_sqrt_hann_is_sine() -> None:
    n = np.arange(512)
    np.testing.assert_allclose(sqrt_hann(512), np.sin(np.pi * n / 512), atol=1e-15)


def test_squared_window_satisfies_cola_at_half_overlap() -> None:
    w2 = sqrt_hann(512) ** 2
    np.testing.assert_allclose(w2[:256] + w2[256:], np.ones(256), atol=1e-14)


def test_sine_window_metrics() -> None:
    metrics = window_metrics(sqrt_hann(512))
    # Exact discrete value: sum(w) = cot(pi / 2N), sum(w^2) = N / 2, so
    # ENBW = (N^2 / 2) tan^2(pi / 2N), which tends to pi^2 / 8 for large N.
    exact = 512**2 / 2.0 * math.tan(math.pi / 1024) ** 2
    assert metrics.enbw_bins == pytest.approx(exact, abs=1e-9)
    assert metrics.enbw_bins == pytest.approx(math.pi**2 / 8, abs=1e-4)
    assert metrics.mainlobe_null_to_null_bins == pytest.approx(3.0, abs=0.05)
    assert metrics.peak_sidelobe_db == pytest.approx(-23.0, abs=0.5)


def test_hann_window_metrics() -> None:
    metrics = window_metrics(hann_periodic(512))
    assert metrics.enbw_bins == pytest.approx(1.5, abs=1e-9)
    assert metrics.mainlobe_null_to_null_bins == pytest.approx(4.0, abs=0.05)
    assert metrics.peak_sidelobe_db == pytest.approx(-31.5, abs=0.5)


def test_rectangular_window_enbw_is_one_bin() -> None:
    assert window_metrics(np.ones(256)).enbw_bins == pytest.approx(1.0)


@pytest.mark.parametrize("bad_length", [0, 1])
def test_window_constructors_reject_tiny_lengths(bad_length: int) -> None:
    with pytest.raises(ValueError):
        hann_periodic(bad_length)
    with pytest.raises(ValueError):
        sqrt_hann(bad_length)


def test_window_metrics_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        window_metrics(np.ones(1))
    with pytest.raises(ValueError):
        window_metrics(np.ones(16), oversample=1)
