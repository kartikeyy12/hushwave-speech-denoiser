"""STFT / ISTFT: perfect reconstruction, framing, and edge cases."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hushwave.stft import istft, num_frames, stft
from hushwave.windows import hann_periodic, sqrt_hann

N, H = 512, 256


@pytest.mark.parametrize("length", [1, 2, 255, 256, 257, 511, 512, 513, 1000, 4096, 12345])
def test_perfect_reconstruction(rng: np.random.Generator, length: int) -> None:
    x = rng.standard_normal(length)
    y = istft(stft(x), length)
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-12


def test_perfect_reconstruction_very_long_input(rng: np.random.Generator) -> None:
    x = rng.standard_normal(16_000 * 60)
    assert np.max(np.abs(istft(stft(x), x.size) - x)) < 1e-12


def test_frame_count_formula(rng: np.random.Generator) -> None:
    for length in (1, 255, 256, 257, 1000):
        assert stft(rng.standard_normal(length)).shape == (num_frames(length, N, H), N // 2 + 1)
    assert num_frames(1, N, H) == 2
    assert num_frames(256, N, H) == 2
    assert num_frames(257, N, H) == 3


def test_stft_matches_independent_numpy_construction(rng: np.random.Generator) -> None:
    x = rng.standard_normal(2000)
    window = sqrt_hann(N)
    frames = num_frames(x.size, N, H)
    padded = np.zeros((frames - 1) * H + N)
    padded[H : H + x.size] = x
    expected = np.stack([np.fft.rfft(padded[f * H : f * H + N] * window) for f in range(frames)])
    assert np.max(np.abs(stft(x) - expected)) < 1e-9


def test_custom_window_round_trip_needs_cola(rng: np.random.Generator) -> None:
    x = rng.standard_normal(3000)
    hann = hann_periodic(N)
    assert np.max(np.abs(istft(stft(x, window=hann), x.size, window=hann) - x)) > 1e-3


def test_all_zero_input_stays_exactly_zero() -> None:
    x = np.zeros(1500)
    assert not np.any(stft(x))
    assert not np.any(istft(stft(x), x.size))


def test_dc_input_is_reconstructed(rng: np.random.Generator) -> None:
    x = np.full(3000, 0.25)
    assert np.max(np.abs(istft(stft(x), x.size) - x)) < 1e-12


def test_clipped_input_is_reconstructed(rng: np.random.Generator) -> None:
    x = np.clip(3.0 * rng.standard_normal(4000), -1.0, 1.0)
    assert np.max(np.abs(istft(stft(x), x.size) - x)) < 1e-12


def test_other_frame_lengths_round_trip(rng: np.random.Generator) -> None:
    x = rng.standard_normal(5000)
    for frame_len in (256, 1024):
        hop = frame_len // 2
        assert np.max(np.abs(istft(stft(x, frame_len, hop), x.size, frame_len, hop) - x)) < 1e-12


def test_interior_stft_is_linear(rng: np.random.Generator) -> None:
    a, b = rng.standard_normal(2000), rng.standard_normal(2000)
    assert np.max(np.abs(stft(2.0 * a - 3.0 * b) - (2.0 * stft(a) - 3.0 * stft(b)))) < 1e-9


def test_output_never_contains_nan_or_inf(rng: np.random.Generator) -> None:
    spec = stft(rng.standard_normal(2000))
    assert np.all(np.isfinite(spec))
    assert np.all(np.isfinite(istft(spec * 0.3, 2000)))


@pytest.mark.parametrize(
    "bad",
    [np.array([]), np.array([1.0, np.nan]), np.array([1.0, np.inf]), np.zeros((2, 100))],
)
def test_invalid_signals_are_rejected(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        stft(bad)


def test_invalid_framing_is_rejected() -> None:
    with pytest.raises(ValueError):
        stft(np.zeros(100), frame_len=500, hop=250)
    with pytest.raises(ValueError):
        stft(np.zeros(100), frame_len=512, hop=128)
    with pytest.raises(ValueError):
        stft(np.zeros(100), window=np.ones(100))
    with pytest.raises(ValueError):
        num_frames(0, N, H)


def test_istft_rejects_inconsistent_shapes() -> None:
    spec = stft(np.zeros(1000))
    with pytest.raises(ValueError):
        istft(spec[:, :10], 1000)
    with pytest.raises(ValueError):
        istft(spec, 5000)
    with pytest.raises(ValueError):
        istft(spec, 0)
    with pytest.raises(ValueError):
        istft(spec, 1000, window=np.ones(10))
    with pytest.raises(ValueError):
        istft(np.zeros(5), 1000)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=1500)
)
def test_round_trip_property(values: list[float]) -> None:
    x = np.asarray(values, dtype=np.float64)
    assert np.max(np.abs(istft(stft(x), x.size) - x)) < 1e-12
