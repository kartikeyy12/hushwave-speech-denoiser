"""Noise PSD trackers: window logic, calibration, convergence, step response, start-up."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from hushwave.config import DEFAULT
from hushwave.noise_est import (
    KINDS,
    McraEstimator,
    MinStatsEstimator,
    SlidingMin,
    calibrated_bias,
    calibrated_bias_curve,
    estimate_noise_psd,
    make_estimator,
    measure_step_delay,
)
from hushwave.stft import stft
from hushwave.synth import pink_noise
from hushwave.windows import sqrt_hann

FS = 16_000
N_BINS = DEFAULT.n_bins
LAMBDA_PER_UNIT_VARIANCE = float(np.sum(sqrt_hann(DEFAULT.frame_len) ** 2))
WINDOW = DEFAULT.ms_window_frames


def _white_power(seed: int, seconds: float, sigma: float = 1.0) -> np.ndarray:
    x = sigma * np.random.default_rng(seed).standard_normal(int(seconds * FS))
    return np.abs(stft(x)) ** 2


def _interior_ratio(estimate: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return np.asarray(estimate[:, 1:-1].mean(axis=1) / (sigma**2 * LAMBDA_PER_UNIT_VARIANCE))


@pytest.mark.parametrize(("n_sub", "sub_frames"), [(1, 4), (3, 4), (6, 5)])
def test_sliding_min_matches_brute_force(
    rng: np.random.Generator, n_sub: int, sub_frames: int
) -> None:
    data = rng.uniform(0.1, 10.0, size=(100, 3))
    tracker = SlidingMin(3, n_sub, sub_frames)
    for n, row in enumerate(data):
        window_index = n // sub_frames
        ends_window = n % sub_frames == sub_frames - 1
        if n_sub == 1:
            first_window = window_index
        else:
            first_window = window_index - n_sub + (2 if ends_window else 1)
        start = max(0, first_window * sub_frames)
        expected = data[start : n + 1].min(axis=0)
        np.testing.assert_array_equal(tracker.update(row), expected)


def test_sliding_min_validates_and_resets() -> None:
    with pytest.raises(ValueError):
        SlidingMin(3, 0, 4)
    tracker = SlidingMin(2, 2, 2)
    tracker.update(np.array([1.0, 1.0]))
    tracker.reset()
    np.testing.assert_array_equal(tracker.update(np.array([5.0, 6.0])), [5.0, 6.0])


def test_calibrated_factors_have_the_expected_character() -> None:
    assert 1.4 < calibrated_bias("minstats", N_BINS) < 1.9
    assert calibrated_bias("mcra", N_BINS) == pytest.approx(1.0, abs=0.05)
    curve = np.asarray(calibrated_bias_curve(N_BINS, alpha_s=0.9, n_sub=6, sub_frames=16))
    assert curve.size == 2 * WINDOW
    assert curve[0] < curve[WINDOW - 20]
    assert curve[-1] == pytest.approx(calibrated_bias("minstats", N_BINS), rel=0.05)
    assert calibrated_bias("minstats", N_BINS) is calibrated_bias("minstats", N_BINS)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("seed", "sigma"), [(101, 1.0), (202, 0.37), (303, 12.0)])
def test_unbiased_on_stationary_white_noise(kind: str, seed: int, sigma: float) -> None:
    """Fresh seeds and levels. Tolerance 5 % (0.21 dB): the calibration run reproduces the mean to
    about 0.2 %, the residual spread of single 100 s runs measured below 3 %."""
    estimate = estimate_noise_psd(_white_power(seed, 100.0, sigma), kind)
    ratio = _interior_ratio(estimate, sigma)
    assert abs(float(ratio[WINDOW:].mean()) - 1.0) < 0.05


@pytest.mark.parametrize("kind", KINDS)
def test_unbiased_on_coloured_noise(kind: str) -> None:
    """Pink noise: compare against the long-time mean periodogram in the speech band."""
    x = 0.2 * pink_noise(FS * 100, np.random.default_rng(9))
    power = np.abs(stft(x)) ** 2
    reference = power.mean(axis=0)
    estimate = estimate_noise_psd(power, kind)[WINDOW:]
    freqs = np.arange(N_BINS) * DEFAULT.bin_hz
    band = (freqs >= 100.0) & (freqs <= 4000.0)
    assert abs(float(np.mean(estimate[:, band] / reference[band])) - 1.0) < 0.05


@pytest.mark.parametrize("kind", KINDS)
def test_converges_quickly_on_stationary_noise(kind: str) -> None:
    ratios = np.mean(
        [_interior_ratio(estimate_noise_psd(_white_power(s, 8.0), kind)) for s in range(500, 512)],
        axis=0,
    )
    assert np.all(np.abs(ratios[30:] - 1.0) < 0.06)
    if kind == "minstats":
        assert np.all(np.abs(ratios[3:] - 1.0) < 0.06)


def test_step_up_is_followed_within_one_window_plus_smoothing() -> None:
    """Minimum statistics must wait for old low values to leave its window; bound = D + 20."""
    for kind in KINDS:
        for ratio_db in (6.0, 10.0):
            assert measure_step_delay(kind, ratio_db).delay_frames <= WINDOW + 20


def test_step_down_is_followed_quickly() -> None:
    for kind in KINDS:
        assert measure_step_delay(kind, -6.0).delay_frames <= 40


def test_step_response_is_deterministic() -> None:
    assert measure_step_delay("mcra", 6.0) == measure_step_delay("mcra", 6.0)


def _speech_like_from_the_first_sample(seed: int = 5) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = FS * 14
    t = np.arange(n) / FS
    speech = np.zeros(n)
    vibrato = 15.0 * np.sin(2.0 * np.pi * 0.7 * t)
    for harmonic in range(1, 26):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        speech += np.sin(2.0 * np.pi * (110.0 * harmonic + vibrato) * t + phase) / harmonic**0.5
    envelope = ((t % 1.6) < 0.6).astype(np.float64)
    sigma = 0.05
    noisy = 0.2 * speech * envelope + sigma * rng.standard_normal(n)
    return np.abs(stft(noisy)) ** 2, sigma**2 * LAMBDA_PER_UNIT_VARIANCE


@pytest.mark.parametrize("kind", KINDS)
def test_recovers_when_speech_is_present_from_the_first_frame(kind: str) -> None:
    """No noise-only start is assumed: speech starts at sample 0, yet within about five seconds
    the band-median estimate settles within +4 / -1 dB of the true noise level."""
    power, true_psd = _speech_like_from_the_first_sample()
    estimate = estimate_noise_psd(power, kind)
    freqs = np.arange(N_BINS) * DEFAULT.bin_hz
    band = (freqs >= 100.0) & (freqs <= 4000.0)
    errors = [
        float(np.median(10.0 * np.log10(estimate[i][band] / true_psd)))
        for i in range(int(5 * FS / DEFAULT.hop), estimate.shape[0])
    ]
    assert min(errors) > -1.0 and max(errors) < 4.0
    start_error = float(np.median(10.0 * np.log10(estimate[10][band] / true_psd)))
    assert np.isfinite(start_error)


@pytest.mark.parametrize("kind", KINDS)
def test_streaming_equals_batch_and_reset_restores_state(kind: str) -> None:
    power = _white_power(77, 3.0)
    batch = estimate_noise_psd(power, kind)
    estimator = make_estimator(kind, N_BINS)
    streamed = np.stack([estimator.update(row) for row in power])
    np.testing.assert_array_equal(streamed, batch)
    estimator.reset()
    np.testing.assert_array_equal(estimator.update(power[0]), batch[0])


@pytest.mark.parametrize("kind", KINDS)
def test_degenerate_inputs_stay_finite(kind: str) -> None:
    zeros = estimate_noise_psd(np.zeros((50, N_BINS)), kind)
    assert np.all(zeros == DEFAULT.noise_floor)
    huge = estimate_noise_psd(np.full((50, N_BINS), 1e30), kind)
    assert np.all(np.isfinite(huge)) and np.all(huge > 0)
    extreme = np.full((200, N_BINS), 1.0)
    extreme[100] = 1e12
    assert np.all(np.isfinite(estimate_noise_psd(extreme, kind)))


@pytest.mark.parametrize("kind", KINDS)
def test_a_forty_db_burst_is_forgotten_within_a_window(kind: str) -> None:
    """A single 1e4-fold power spike must not leave the estimate more than 3x too high 100
    frames later (exponential smoothing needs about 100 frames to forget a 40 dB burst)."""
    frames = np.full((220, N_BINS), 1.0)
    frames[100] = 1e4
    out = estimate_noise_psd(frames, kind)
    baseline = out[90].mean()
    assert out[210].mean() < 3.0 * baseline


@pytest.mark.parametrize("kind", KINDS)
def test_scale_equivariance(kind: str) -> None:
    power = _white_power(5, 4.0)
    base = estimate_noise_psd(power, kind)
    scaled = estimate_noise_psd(1e4 * power, kind)
    np.testing.assert_allclose(scaled, 1e4 * base, rtol=1e-9)


_frames = hnp.arrays(
    np.float64,
    st.tuples(st.integers(1, 40), st.just(9)),
    elements=st.floats(0.0, 1e12, allow_nan=False, allow_infinity=False),
)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_frames)
def test_output_is_always_finite_and_at_least_the_floor(frames: np.ndarray) -> None:
    for kind in KINDS:
        estimate = estimate_noise_psd(frames, kind)
        assert estimate.shape == frames.shape
        assert np.all(np.isfinite(estimate))
        assert np.all(estimate >= DEFAULT.noise_floor)


def test_invalid_inputs_and_parameters_are_rejected() -> None:
    estimator = MinStatsEstimator(N_BINS)
    with pytest.raises(ValueError, match="shape"):
        estimator.update(np.ones(10))
    with pytest.raises(ValueError, match="non-negative"):
        estimator.update(-np.ones(N_BINS))
    with pytest.raises(ValueError, match="finite"):
        estimator.update(np.full(N_BINS, np.nan))
    with pytest.raises(ValueError):
        MinStatsEstimator(1)
    with pytest.raises(ValueError):
        MinStatsEstimator(N_BINS, alpha_s=1.0)
    with pytest.raises(ValueError):
        McraEstimator(1)
    with pytest.raises(ValueError):
        McraEstimator(N_BINS, alpha_p=0.0)
    with pytest.raises(ValueError):
        McraEstimator(N_BINS, delta=1.0)
    with pytest.raises(ValueError, match="unknown estimator"):
        make_estimator("kalman", N_BINS)
    with pytest.raises(ValueError):
        estimate_noise_psd(np.ones(5), "mcra")


def test_constant_bias_override_is_applied() -> None:
    power = _white_power(3, 2.0)
    free = estimate_noise_psd(power, "mcra", bias=1.0)
    doubled = estimate_noise_psd(power, "mcra", bias=2.0)
    np.testing.assert_allclose(doubled, 2.0 * free)
    assert MinStatsEstimator(N_BINS, bias=3.0).bias == 3.0
    assert make_estimator("minstats", N_BINS).bias == pytest.approx(
        calibrated_bias("minstats", N_BINS)
    )


def test_parameter_overrides_reach_the_estimators() -> None:
    power = _white_power(4, 2.0)
    default = estimate_noise_psd(power, "minstats")
    smoother = estimate_noise_psd(power, "minstats", alpha_s=0.7)
    assert not np.allclose(default, smoother)
    stricter = estimate_noise_psd(power, "mcra", delta=2.0)
    assert not np.allclose(stricter, estimate_noise_psd(power, "mcra"))
