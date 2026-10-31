"""Closed-form results and the aliasing demonstration."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import exp1

from hushwave.theory import (
    aliased_frequency_hz,
    aliasing_experiment,
    butterworth_highpass_gain_db,
    quantization_sqnr_db,
)


def test_sqnr_formula_16_bit() -> None:
    assert quantization_sqnr_db(16) == pytest.approx(6.0206 * 16 + 1.7609, abs=1e-3)
    assert quantization_sqnr_db(16) == pytest.approx(98.09, abs=0.01)


def test_sqnr_level_offset_is_db_for_db() -> None:
    assert quantization_sqnr_db(16, 40.0) == pytest.approx(quantization_sqnr_db(16) - 40.0)


def test_sqnr_each_bit_adds_six_db() -> None:
    assert quantization_sqnr_db(12) - quantization_sqnr_db(11) == pytest.approx(6.0206, abs=1e-3)


def test_sqnr_matches_measured_rounding_error() -> None:
    n = np.arange(1 << 16, dtype=np.float64)
    x = np.sin(2.0 * np.pi * 0.0123456789 * n) * (1.0 - 2.0**-15)
    scale = 2.0**15
    quantized = np.clip(np.round(x * scale), -scale, scale - 1.0) / scale
    measured = 10.0 * np.log10(np.mean(x**2) / np.mean((x - quantized) ** 2))
    assert measured == pytest.approx(quantization_sqnr_db(16), abs=0.5)


@pytest.mark.parametrize(
    ("freq", "fs", "expected"),
    [
        (11_000.0, 16_000.0, 5_000.0),
        (3_000.0, 16_000.0, 3_000.0),
        (8_000.0, 16_000.0, 8_000.0),
        (16_000.0, 16_000.0, 0.0),
        (17_000.0, 16_000.0, 1_000.0),
        (0.0, 16_000.0, 0.0),
    ],
)
def test_aliased_frequency(freq: float, fs: float, expected: float) -> None:
    assert aliased_frequency_hz(freq, fs) == pytest.approx(expected)


def test_aliased_frequency_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        aliased_frequency_hz(-1.0, 16_000.0)
    with pytest.raises(ValueError):
        aliased_frequency_hz(100.0, 0.0)


def test_highpass_gain_at_cutoff_is_minus_three_db() -> None:
    assert butterworth_highpass_gain_db(80.0, 80.0, 16_000.0) == pytest.approx(
        -10.0 * math.log10(2.0)
    )


def test_highpass_gain_is_monotone_and_flat_in_passband() -> None:
    gains = [butterworth_highpass_gain_db(f, 80.0, 16_000.0) for f in (10, 40, 80, 160, 1_000)]
    assert gains == sorted(gains)
    assert gains[-1] == pytest.approx(0.0, abs=1e-3)


def test_highpass_gain_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        butterworth_highpass_gain_db(0.0, 80.0, 16_000.0)
    with pytest.raises(ValueError):
        butterworth_highpass_gain_db(100.0, 9_000.0, 16_000.0)


def test_sqnr_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        quantization_sqnr_db(0)
    with pytest.raises(ValueError):
        quantization_sqnr_db(16, -1.0)


def test_aliasing_without_filter_is_full_strength() -> None:
    result = aliasing_experiment()
    assert result.predicted_alias_hz == pytest.approx(5_000.0)
    assert result.naive_alias_db == pytest.approx(0.0, abs=1.0)


def test_aliasing_with_filter_is_strongly_suppressed() -> None:
    # A Hamming-window FIR has a stop-band floor near -53 dB; require at least 40 dB.
    result = aliasing_experiment()
    assert result.filtered_alias_db < -40.0


def test_aliasing_experiment_is_deterministic() -> None:
    first, second = aliasing_experiment(), aliasing_experiment()
    np.testing.assert_array_equal(first.naive_spectrum_db, second.naive_spectrum_db)
    np.testing.assert_array_equal(first.filtered_spectrum_db, second.filtered_spectrum_db)


def test_aliasing_experiment_rejects_non_integer_factor() -> None:
    with pytest.raises(ValueError):
        aliasing_experiment(source_fs=48_000, factor=7)


def test_log_moment_identity_used_for_mmse_lsa(rng: np.random.Generator) -> None:
    """E[ln |m + Z|^2] = ln |m|^2 + E1(|m|^2) for Z ~ CN(0, 1) (THEORY.md section 7.3)."""
    z = (rng.standard_normal(400_000) + 1j * rng.standard_normal(400_000)) / math.sqrt(2.0)
    for m_squared in (0.05, 0.5, 2.0, 10.0):
        measured = float(np.mean(np.log(np.abs(math.sqrt(m_squared) + z) ** 2)))
        assert measured == pytest.approx(math.log(m_squared) + float(exp1(m_squared)), abs=0.02)


def test_raw_lsa_gain_can_exceed_one() -> None:
    xi, gamma = 1.0, 0.01
    v = xi / (1.0 + xi) * gamma
    assert xi / (1.0 + xi) * math.exp(0.5 * float(exp1(v))) > 1.0
