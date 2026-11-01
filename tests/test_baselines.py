"""Baselines B0 to B4 and the shared offline spectral pipeline."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hushwave.baselines import (
    BASELINE_NAMES,
    b0_unprocessed,
    b1_spectral_subtraction,
    b2_scipy_wiener,
    b3_noisereduce,
    b4_plain_wiener,
    run_baseline,
    spectral_denoise,
)
from hushwave.config import DEFAULT, Config
from hushwave.gains import RULES

FS = DEFAULT.fs


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    return float(10.0 * np.log10(np.sum(clean**2) / np.sum((estimate - clean) ** 2)))


def speech_like(
    rng: np.random.Generator, seconds: float = 6.0, snr: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic bursts (0.5 s on / 0.5 s off) in white noise at exactly `snr` dB."""
    n = round(seconds * FS)
    t = np.arange(n) / FS
    voiced = sum(np.sin(2.0 * np.pi * 150.0 * k * t) / k for k in range(1, 12))
    gate = (np.sin(2.0 * np.pi * t) > 0).astype(np.float64)
    gate = np.convolve(gate, np.hanning(800) / np.hanning(800).sum(), mode="same")
    clean = 0.3 * voiced * gate
    noise = rng.standard_normal(n)
    scale = np.sqrt(np.sum(clean**2) / (np.sum(noise**2) * 10.0 ** (snr / 10.0)))
    return clean, clean + scale * noise


# ------------------------------------------------------------ shared pipeline
@pytest.mark.parametrize("rule", RULES)
def test_unit_floor_makes_the_pipeline_transparent(rule: str, rng: np.random.Generator) -> None:
    x = rng.standard_normal(5000)
    out = spectral_denoise(x, rule, gain_overrides={"g_min": 1.0})
    np.testing.assert_allclose(out, x, atol=1e-10)


@pytest.mark.parametrize("length", [1, 255, 256, 257, 3001])
def test_output_length_matches_input(length: int, rng: np.random.Generator) -> None:
    x = rng.standard_normal(length)
    assert spectral_denoise(x, "lsa").shape == x.shape


@pytest.mark.parametrize("estimator", ["mcra", "minstats"])
def test_silence_stays_silent_and_finite(estimator: str) -> None:
    out = spectral_denoise(np.zeros(4000), "lsa", estimator)
    assert np.all(out == 0.0)


@pytest.mark.parametrize(
    ("rule", "limit_db"),
    [("subtraction", -15.0), ("wiener", -15.0), ("lsa", -12.0), ("ml_wiener", -6.0)],
)
def test_noise_only_input_is_attenuated(
    rule: str, limit_db: float, rng: np.random.Generator
) -> None:
    noise = 0.05 * rng.standard_normal(5 * FS)
    out = spectral_denoise(noise, rule)
    change = 10.0 * np.log10(np.mean(out[FS:] ** 2) / np.mean(noise[FS:] ** 2))
    assert change < limit_db


@pytest.mark.parametrize(
    ("rule", "minimum_gain_db"),
    [("subtraction", 1.0), ("wiener", 3.0), ("lsa", 3.0), ("ml_wiener", 3.0)],
)
def test_speech_in_noise_improves_snr(
    rule: str, minimum_gain_db: float, rng: np.random.Generator
) -> None:
    clean, noisy = speech_like(rng)
    out = spectral_denoise(noisy, rule)
    tail = slice(3 * FS, None)
    assert snr_db(clean[tail], out[tail]) - snr_db(clean[tail], noisy[tail]) > minimum_gain_db


def test_both_estimators_are_accepted(rng: np.random.Generator) -> None:
    x = rng.standard_normal(4000)
    a = spectral_denoise(x, "wiener", "mcra")
    b = spectral_denoise(x, "wiener", "minstats")
    assert a.shape == b.shape == x.shape
    assert not np.allclose(a, b)


def test_pipeline_uses_the_given_config(rng: np.random.Generator) -> None:
    cfg = Config(frame_len=256, hop=128)
    x = rng.standard_normal(4000)
    assert spectral_denoise(x, "lsa", cfg=cfg).shape == x.shape


def test_overrides_reach_the_estimator_and_the_gain_stage(rng: np.random.Generator) -> None:
    x = 0.1 * rng.standard_normal(6000)
    base = spectral_denoise(x, "lsa")
    assert not np.allclose(base, spectral_denoise(x, "lsa", gain_overrides={"g_min": 0.5}))
    assert not np.allclose(base, spectral_denoise(x, "lsa", estimator_overrides={"alpha_s": 0.5}))


@pytest.mark.parametrize("bad", [np.ones((4, 4)), np.array([]), np.array([1.0, np.nan])])
def test_inputs_are_validated(bad: np.ndarray) -> None:
    for function in (b0_unprocessed, b2_scipy_wiener, spectral_denoise_lsa):
        with pytest.raises(ValueError):
            function(bad)


def spectral_denoise_lsa(x: np.ndarray) -> np.ndarray:
    return spectral_denoise(x, "lsa")


# ------------------------------------------------------------ individual baselines
def test_b0_returns_an_equal_copy(rng: np.random.Generator) -> None:
    x = rng.standard_normal(100)
    out = b0_unprocessed(x)
    np.testing.assert_array_equal(out, x)
    out[0] = 99.0
    assert x[0] != 99.0


def test_b1_and_b4_disable_temporal_smoothing(rng: np.random.Generator) -> None:
    x = 0.1 * rng.standard_normal(6000)
    np.testing.assert_array_equal(
        b1_spectral_subtraction(x),
        spectral_denoise(x, "subtraction", gain_overrides={"gain_mu": 0}),
    )
    np.testing.assert_array_equal(
        b4_plain_wiener(x), spectral_denoise(x, "ml_wiener", gain_overrides={"gain_mu": 0})
    )


def test_b2_reduces_white_noise_and_keeps_shape(rng: np.random.Generator) -> None:
    x = rng.standard_normal(4000)
    out = b2_scipy_wiener(x)
    assert out.shape == x.shape
    assert np.mean(out**2) < 0.5 * np.mean(x**2)


def test_b2_silence_and_window_validation() -> None:
    assert np.all(b2_scipy_wiener(np.zeros(100)) == 0.0)
    for window in (1, 4, 2):
        with pytest.raises(ValueError, match="odd"):
            b2_scipy_wiener(np.ones(100), window=window)


def test_b3_reports_a_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "noisereduce", None)
    with pytest.raises(ImportError, match="pip install"):
        b3_noisereduce(np.ones(100))


def _fake_noisereduce(
    monkeypatch: pytest.MonkeyPatch, result: np.ndarray
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def reduce_noise(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return result

    module = types.ModuleType("noisereduce")
    module.reduce_noise = reduce_noise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "noisereduce", module)
    return calls


def test_b3_calls_noisereduce_non_stationary_and_casts_to_float64(
    monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator
) -> None:
    x = rng.standard_normal(500)
    calls = _fake_noisereduce(monkeypatch, (0.5 * x).astype(np.float32))
    out = b3_noisereduce(x, FS)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, 0.5 * x, rtol=1e-6)
    assert calls[0]["sr"] == FS
    assert calls[0]["stationary"] is False


def test_b3_rejects_a_wrong_output_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_noisereduce(monkeypatch, np.zeros(7))
    with pytest.raises(RuntimeError, match="shape"):
        b3_noisereduce(np.ones(100))


def test_b3_silence_is_returned_without_calling_the_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_noisereduce(monkeypatch, np.zeros(100))
    assert np.all(b3_noisereduce(np.zeros(100)) == 0.0)
    assert calls == []


def test_b3_with_the_real_package(rng: np.random.Generator) -> None:
    pytest.importorskip("noisereduce")
    x = 0.1 * rng.standard_normal(2 * FS)
    out = b3_noisereduce(x, FS)
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


# ------------------------------------------------------------ registry
def test_run_baseline_dispatches_every_name(
    monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator
) -> None:
    x = 0.1 * rng.standard_normal(4000)
    _fake_noisereduce(monkeypatch, 0.5 * x)
    for name in BASELINE_NAMES:
        out = run_baseline(name, x)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(run_baseline("B0", x), x)
    np.testing.assert_array_equal(run_baseline("B4", x), b4_plain_wiener(x))
    np.testing.assert_array_equal(
        run_baseline("B1", x, estimator="minstats"),
        b1_spectral_subtraction(x, estimator="minstats"),
    )


def test_run_baseline_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown baseline"):
        run_baseline("B9", np.ones(100))
