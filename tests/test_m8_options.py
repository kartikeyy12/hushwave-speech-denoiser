"""The opt-in options added in M8: spectral high-pass, rise limiter, mask-based metrics."""

from __future__ import annotations

import numpy as np
import pytest

from hushwave.config import DEFAULT, Config
from hushwave.enhancer import enhance, stream
from hushwave.filters import frequency_response, highpass_biquad
from hushwave.gains import GainStage, spectral_highpass_mask
from hushwave.metrics import (
    musical_noise_masked,
    musical_noise_ratio,
    noise_reduction_db,
    noise_reduction_masked,
    speech_activity,
)
from hushwave.noise_est import RiseLimitedEstimator, make_estimator
from hushwave.stft import stft

N_BINS = DEFAULT.n_bins


def test_mask_matches_the_biquad_magnitude_at_bin_centres() -> None:
    mask = spectral_highpass_mask(N_BINS, DEFAULT.hp_cutoff_hz, DEFAULT.fs)
    b, a = highpass_biquad(DEFAULT.hp_cutoff_hz, DEFAULT.fs)
    freqs = np.arange(1, N_BINS - 1) * DEFAULT.bin_hz
    reference = np.abs(frequency_response(b, a, freqs, DEFAULT.fs))
    np.testing.assert_allclose(mask[1:-1], reference, rtol=1e-9)
    assert mask[0] == 0.0
    assert mask[-1] == 1.0
    assert np.all(np.diff(mask) >= 0.0)


def test_mask_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        spectral_highpass_mask(1, 80.0, 16000.0)
    with pytest.raises(ValueError):
        spectral_highpass_mask(257, 8000.0, 16000.0)


def test_gain_stage_applies_the_mask_after_the_recursion(rng: np.random.Generator) -> None:
    plain = GainStage("lsa", N_BINS)
    masked = GainStage("lsa", N_BINS, spectral_hp=1.0)
    mask = spectral_highpass_mask(N_BINS, DEFAULT.hp_cutoff_hz, DEFAULT.fs)
    for _ in range(12):
        power = rng.exponential(2.0, N_BINS)
        noise = np.full(N_BINS, 1.0)
        np.testing.assert_allclose(masked.process(power, noise), plain.process(power, noise) * mask)
    assert GainStage("lsa", N_BINS, Config(spectral_hp=True))._hp_mask is not None
    assert GainStage("lsa", N_BINS, spectral_hp=0.0)._hp_mask is None


def test_streaming_equals_batch_with_both_options(rng: np.random.Generator) -> None:
    signal = 0.1 * rng.standard_normal(16000)
    options = {
        "gain_overrides": {"spectral_hp": 1.0},
        "estimator_overrides": {"rise_db_per_s": 3.0},
        "prefilter": False,
    }
    batch = enhance(signal, **options)  # type: ignore[arg-type]
    streamed = stream(signal, [1, 256, 777, 53], **options)  # type: ignore[arg-type]
    np.testing.assert_allclose(streamed, batch, atol=1e-9)


def test_spectral_highpass_has_no_phase_error() -> None:
    """A low tone passes through with the mask's gain and zero delay (the biquad shifts it)."""
    t = np.arange(16000) / 16000.0
    tone = np.sin(2 * np.pi * 125.0 * t)
    out = enhance(tone, prefilter=False, gain_overrides={"spectral_hp": 1.0, "g_min": 1.0})
    core = slice(2000, 14000)
    gain = float(np.dot(out[core], tone[core]) / np.dot(tone[core], tone[core]))
    residual = out[core] - gain * tone[core]
    assert np.sqrt(np.mean(residual**2)) < 0.02 * np.sqrt(np.mean(tone[core] ** 2))
    assert 0.9 < gain <= 1.0


@pytest.mark.parametrize("kind", ["minstats", "mcra"])
def test_rise_limiter_caps_the_rise_and_not_the_fall(kind: str) -> None:
    est = make_estimator(kind, N_BINS, rise_db_per_s=6.0)
    assert isinstance(est, RiseLimitedEstimator)
    frames = np.concatenate([np.full((120, N_BINS), 1.0), np.full((120, N_BINS), 100.0)])
    out = np.stack([est.update(f) for f in frames])
    assert np.all(out[1:] <= out[:-1] * est.ratio * (1.0 + 1e-12))
    assert out[-1].mean() < 100.0 * 0.5
    est.reset()
    assert est.update(np.full(N_BINS, 100.0)).mean() > out[130].mean()
    low = make_estimator(kind, N_BINS, rise_db_per_s=6.0)
    drop = np.stack([low.update(np.full(N_BINS, v)) for v in [100.0] * 100 + [1.0] * 10])
    assert drop[-1].mean() < drop[99].mean()


def test_rise_limiter_is_off_by_default_and_validated() -> None:
    assert not isinstance(make_estimator("mcra", N_BINS), RiseLimitedEstimator)
    assert make_estimator("mcra", N_BINS, Config(rise_db_per_s=3.0)).ratio > 1.0  # type: ignore[union-attr]
    inner = make_estimator("minstats", N_BINS)
    with pytest.raises(ValueError):
        RiseLimitedEstimator(inner, 0.0, 0.016)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RiseLimitedEstimator(inner, 3.0, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Config(rise_db_per_s=-1.0)
    limited = make_estimator("minstats", N_BINS, rise_db_per_s=3.0)
    assert limited.bias == pytest.approx(inner.bias)


def test_rise_limiter_keeps_stationary_noise_nearly_unbiased(rng: np.random.Generator) -> None:
    noise = rng.standard_normal(16000 * 12)
    power = np.abs(stft(noise, DEFAULT.frame_len, DEFAULT.hop)) ** 2
    for kind in ("minstats", "mcra"):
        free = make_estimator(kind, N_BINS)
        capped = make_estimator(kind, N_BINS, rise_db_per_s=3.0)
        a = np.stack([free.update(p) for p in power])[300:].mean()
        b = np.stack([capped.update(p) for p in power])[300:].mean()
        assert abs(10 * np.log10(b / a)) < 1.0


def test_masked_metrics_equal_the_clean_reference_versions(rng: np.random.Generator) -> None:
    n = 16000 * 4
    clean = np.zeros(n)
    clean[16000:48000] = 0.3 * rng.standard_normal(32000)
    noisy = clean + 0.05 * rng.standard_normal(n)
    enhanced = 0.4 * noisy
    quiet = ~speech_activity(clean)
    assert noise_reduction_masked(noisy, enhanced, quiet) == pytest.approx(
        noise_reduction_db(clean, noisy, enhanced)
    )
    assert musical_noise_masked(noisy, enhanced, quiet) == pytest.approx(
        musical_noise_ratio(clean, noisy, enhanced)
    )
    assert np.isnan(noise_reduction_masked(noisy, enhanced, np.zeros_like(quiet)))
    assert np.isnan(musical_noise_masked(noisy, enhanced, np.zeros_like(quiet)))
    assert np.isnan(noise_reduction_masked(np.zeros(n), enhanced, quiet))
