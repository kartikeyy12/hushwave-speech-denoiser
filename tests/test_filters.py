"""Our biquad against scipy.signal, plus state handling and stability."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.signal import butter, lfilter

from hushwave.filters import Biquad, frequency_response, highpass_biquad, pole_radii

FS = 16_000.0


def test_coefficients_match_scipy_butter() -> None:
    for cutoff in (20.0, 80.0, 300.0, 1000.0):
        b, a = highpass_biquad(cutoff, FS)
        b_ref, a_ref = butter(2, cutoff, btype="highpass", fs=FS)
        assert np.max(np.abs(b - b_ref)) < 1e-12
        assert np.max(np.abs(a - a_ref)) < 1e-12


def test_filtering_matches_lfilter(rng: np.random.Generator) -> None:
    b, a = highpass_biquad(80.0, FS)
    b_ref, a_ref = butter(2, 80.0, btype="highpass", fs=FS)
    x = rng.standard_normal(20_000)
    assert np.max(np.abs(Biquad(b, a).process(x) - lfilter(b_ref, a_ref, x))) < 1e-9


def test_block_processing_equals_one_shot(rng: np.random.Generator) -> None:
    b, a = highpass_biquad(80.0, FS)
    x = rng.standard_normal(5_000)
    whole = Biquad(b, a).process(x)
    stream = Biquad(b, a)
    pieces = [stream.process(chunk) for chunk in np.array_split(x, [1, 129, 130, 1_000, 4_999])]
    assert np.max(np.abs(np.concatenate(pieces) - whole)) < 1e-12


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(1, 700), min_size=1, max_size=12))
def test_any_block_split_equals_one_shot_property(sizes: list[int]) -> None:
    rng = np.random.default_rng(sum(sizes))
    x = rng.standard_normal(sum(sizes))
    b, a = highpass_biquad(80.0, FS)
    whole = Biquad(b, a).process(x)
    stream = Biquad(b, a)
    out = []
    start = 0
    for size in sizes:
        out.append(stream.process(x[start : start + size]))
        start += size
    assert np.max(np.abs(np.concatenate(out) - whole)) < 1e-12


def test_reset_restores_initial_state(rng: np.random.Generator) -> None:
    b, a = highpass_biquad(80.0, FS)
    x = rng.standard_normal(500)
    filt = Biquad(b, a)
    first = filt.process(x)
    filt.reset()
    np.testing.assert_array_equal(filt.process(x), first)


def test_gain_at_cutoff_is_minus_three_db() -> None:
    b, a = highpass_biquad(80.0, FS)
    gain = abs(frequency_response(b, a, [80.0], FS)[0])
    assert 20.0 * math.log10(gain) == pytest.approx(-10.0 * math.log10(2.0), abs=1e-9)


def test_frequency_response_matches_scipy_and_theory() -> None:
    b, a = highpass_biquad(80.0, FS)
    freqs = np.array([10.0, 50.0, 80.0, 100.0, 500.0, 4000.0])
    ours = np.abs(frequency_response(b, a, freqs, FS))
    ratio = math.tan(math.pi * 80.0 / FS) / np.tan(np.pi * freqs / FS)
    np.testing.assert_allclose(ours, 1.0 / np.sqrt(1.0 + ratio**4), rtol=1e-9)


def test_dc_is_removed_and_high_frequencies_pass() -> None:
    b, a = highpass_biquad(80.0, FS)
    dc = Biquad(b, a).process(np.ones(16_000))
    assert abs(dc[-1]) < 1e-6
    n = np.arange(16_000)
    tone = np.sin(2.0 * np.pi * 2000.0 * n / FS)
    out = Biquad(b, a).process(tone)
    # A high-pass shifts phase but leaves the amplitude of a 2 kHz tone essentially untouched.
    rms_ratio = np.sqrt(np.mean(out[4000:] ** 2) / np.mean(tone[4000:] ** 2))
    assert rms_ratio == pytest.approx(1.0, abs=1e-3)


def test_poles_are_inside_the_unit_circle() -> None:
    for cutoff in (1.0, 80.0, 2000.0, 7900.0):
        _, a = highpass_biquad(cutoff, FS)
        assert np.all(pole_radii(a) < 1.0)


def test_zero_input_gives_zero_output() -> None:
    b, a = highpass_biquad(80.0, FS)
    assert not np.any(Biquad(b, a).process(np.zeros(100)))


def test_constructor_normalises_a0() -> None:
    b, a = highpass_biquad(80.0, FS)
    scaled = Biquad(2.0 * b, 2.0 * a)
    np.testing.assert_allclose(scaled.a, a, atol=1e-15)
    np.testing.assert_allclose(scaled.b, b, atol=1e-15)


def test_invalid_designs_and_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        highpass_biquad(0.0, FS)
    with pytest.raises(ValueError):
        highpass_biquad(8000.0, FS)
    with pytest.raises(ValueError):
        highpass_biquad(80.0, 0.0)
    with pytest.raises(ValueError):
        Biquad([1.0, 2.0], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        Biquad([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    b, a = highpass_biquad(80.0, FS)
    with pytest.raises(ValueError):
        Biquad(b, a).process(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        Biquad(b, a).process(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        pole_radii([2.0, 0.0, 0.0])
