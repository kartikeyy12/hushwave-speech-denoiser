"""Metrics: every one is checked against a case whose answer is known by construction."""

from __future__ import annotations

import importlib.metadata
import math
import sys
import types
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hushwave.config import DEFAULT, Config
from hushwave.metrics import (
    Metrics,
    active_fraction,
    delta_snr_db,
    evaluate,
    external_versions,
    lsd_db,
    musical_noise_ratio,
    noise_reduction_db,
    pesq_wb,
    segmental_snr_db,
    snr_db,
    speech_activity,
    speech_distortion_db,
    stoi_score,
)
from hushwave.stft import istft, stft
from hushwave.synth import mix_at_snr

FS = DEFAULT.fs
B = DEFAULT.hop  # block length used by the activity labelling


def piecewise(amplitudes: list[float], block: int = B) -> np.ndarray:
    """Signal whose block b has constant magnitude amplitudes[b] and alternating sign."""
    signs = np.where(np.arange(block) % 2 == 0, 1.0, -1.0)
    return np.concatenate([a * signs for a in amplitudes])


def db_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


# ------------------------------------------------------------ waveform SNR
def test_snr_known_answer() -> None:
    x = np.full(1000, 2.0)
    assert snr_db(x, x + 0.2) == pytest.approx(20.0)
    assert snr_db(x, x + 2.0) == pytest.approx(0.0)


def test_snr_of_an_exact_estimate_is_infinite() -> None:
    x = np.array([1.0, -2.0, 3.0])
    assert snr_db(x, x) == math.inf


def test_snr_agrees_with_the_mixing_definition(rng: np.random.Generator) -> None:
    clean = rng.standard_normal(5000)
    noise = rng.standard_normal(5000)
    for target in (-5.0, 0.0, 7.5, 15.0):
        mix = mix_at_snr(clean, noise, target)
        assert snr_db(clean, mix.noisy) == pytest.approx(target, abs=1e-9)


@given(
    scale=st.floats(min_value=1e-3, max_value=1e3),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=50, deadline=None)
def test_snr_is_invariant_to_a_common_gain(scale: float, seed: int) -> None:
    r = np.random.default_rng(seed)
    x, e = r.standard_normal(300), r.standard_normal(300)
    assert snr_db(scale * x, scale * e) == pytest.approx(snr_db(x, e), abs=1e-9)


def test_delta_snr_is_output_minus_input(rng: np.random.Generator) -> None:
    clean = rng.standard_normal(2000)
    noisy = clean + 0.5 * rng.standard_normal(2000)
    enhanced = clean + 0.1 * (noisy - clean)
    assert delta_snr_db(clean, noisy, enhanced) == pytest.approx(20.0, abs=1e-9)


@pytest.mark.parametrize(
    "call",
    [
        lambda: snr_db(np.zeros(10), np.ones(10)),
        lambda: snr_db(np.ones(10), np.ones(9)),
        lambda: snr_db(np.ones(10), np.full(10, np.nan)),
        lambda: snr_db(np.ones((2, 5)), np.ones((2, 5))),
        lambda: snr_db(np.array([]), np.array([])),
    ],
)
def test_snr_rejects_bad_input(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]


# ------------------------------------------------------------ activity labelling
def test_activity_threshold_is_forty_db_below_the_loudest_block() -> None:
    x = piecewise([1.0, db_amp(-39.0), db_amp(-41.0), 1.0, db_amp(-60.0)])
    assert speech_activity(x).tolist() == [True, True, False, True, False]


def test_activity_uses_mean_square_so_a_short_last_block_is_not_penalised() -> None:
    x = np.concatenate([piecewise([1.0, 1.0]), piecewise([db_amp(-39.5)])[:100]])
    assert speech_activity(x).tolist() == [True, True, True]
    y = np.concatenate([piecewise([1.0, 1.0]), piecewise([db_amp(-41.0)])[:100]])
    assert speech_activity(y).tolist() == [True, True, False]


def test_activity_threshold_follows_the_config() -> None:
    x = piecewise([1.0, db_amp(-30.0), 1.0])
    assert speech_activity(x, Config(activity_db=40.0)).tolist() == [True, True, True]
    assert speech_activity(x, Config(activity_db=20.0)).tolist() == [True, False, True]


def test_activity_needs_a_non_silent_reference() -> None:
    with pytest.raises(ValueError, match="all zeros"):
        speech_activity(np.zeros(1000))


def test_active_fraction() -> None:
    assert active_fraction(np.array([True, False, False, True])) == 0.5


# ------------------------------------------------------------ segmental SNR
def test_segmental_snr_clamps_each_block_and_ignores_noise_only_blocks() -> None:
    clean = piecewise([1.0, 1.0, 1.0, 1.0, 1.0, 1e-3])
    targets = [0.0, 10.0, 50.0, -30.0, None]
    error = np.concatenate(
        [piecewise([db_amp(-t)]) if t is not None else np.zeros(B) for t in targets]
        + [piecewise([5.0])]  # huge error inside the noise-only block: must not count
    )
    # clean has alternating signs; the error uses the same sign pattern, so e = clean * ratio
    estimate = clean + error
    expected = np.mean([0.0, 10.0, 35.0, -10.0, 35.0])
    assert segmental_snr_db(clean, estimate) == pytest.approx(expected)


def test_segmental_snr_of_a_perfect_estimate_is_the_upper_clamp() -> None:
    x = piecewise([1.0, 0.5, 1.0])
    assert segmental_snr_db(x, x) == 35.0


@given(
    seed=st.integers(min_value=0, max_value=1000), level=st.floats(min_value=1e-4, max_value=1e3)
)
@settings(max_examples=40, deadline=None)
def test_segmental_snr_always_lies_between_the_clamps(seed: int, level: float) -> None:
    r = np.random.default_rng(seed)
    x = r.standard_normal(3000)
    assert -10.0 <= segmental_snr_db(x, x + level * r.standard_normal(3000)) <= 35.0


# ------------------------------------------------------------ noise reduction and distortion
def test_noise_reduction_known_answer_uses_only_noise_only_blocks() -> None:
    clean = piecewise([1.0, 1e-3, 1e-3, 1.0])
    noisy = piecewise([1.0, 0.05, 0.05, 1.0])
    enhanced = piecewise([7.0, 0.005, 0.005, 7.0])  # changes in active blocks must not matter
    assert noise_reduction_db(clean, noisy, enhanced) == pytest.approx(20.0)


def test_noise_reduction_is_nan_without_noise_only_blocks_or_without_input_noise() -> None:
    loud = piecewise([1.0, 1.0, 1.0])
    assert math.isnan(noise_reduction_db(loud, loud, loud))
    clean = piecewise([1.0, 1e-3])
    assert math.isnan(noise_reduction_db(clean, piecewise([1.0, 0.0]), clean))


def test_speech_distortion_known_answer_uses_only_active_blocks() -> None:
    clean = piecewise([1.0, 1e-3, 1.0])
    speech = clean * 1.1
    speech[B : 2 * B] = 9.0  # wrecked noise-only block: must not count
    assert speech_distortion_db(clean, speech) == pytest.approx(-20.0, abs=1e-9)


def test_speech_distortion_of_untouched_speech_is_very_low_and_finite() -> None:
    clean = piecewise([1.0, 1.0])
    value = speech_distortion_db(clean, clean)
    assert math.isfinite(value)
    assert value < -200.0


# ------------------------------------------------------------ log-spectral distance
def test_lsd_of_identical_signals_is_zero(rng: np.random.Generator) -> None:
    x = rng.standard_normal(6000)
    assert lsd_db(x, x) == 0.0


def test_lsd_of_a_scaled_copy_is_the_scale_in_db(rng: np.random.Generator) -> None:
    x = rng.standard_normal(8000)
    assert lsd_db(x, 0.5 * x) == pytest.approx(20.0 * math.log10(2.0), abs=1e-3)
    assert lsd_db(x, 3.0 * x) == pytest.approx(20.0 * math.log10(3.0), abs=1e-3)


def test_lsd_only_counts_frames_that_touch_speech(rng: np.random.Generator) -> None:
    n = 16000
    clean = np.concatenate([rng.standard_normal(n // 2), 1e-5 * rng.standard_normal(n // 2)])
    estimate = clean.copy()
    estimate[n // 2 + 2 * DEFAULT.frame_len :] = rng.standard_normal(n // 2 - 2 * DEFAULT.frame_len)
    assert lsd_db(clean, estimate) == 0.0
    estimate[100] += 1.0
    assert lsd_db(clean, estimate) > 0.0


def test_lsd_works_on_a_signal_shorter_than_a_frame(rng: np.random.Generator) -> None:
    x = rng.standard_normal(100)
    assert lsd_db(x, 2.0 * x) == pytest.approx(20.0 * math.log10(2.0), abs=1e-3)


# ------------------------------------------------------------ musical noise
def _noise_floor_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = 4 * FS
    clean = 1e-4 * rng.standard_normal(n)
    clean[FS : FS + FS // 2] += 0.5 * rng.standard_normal(FS // 2)
    return clean, clean + 0.1 * rng.standard_normal(n)


def test_musical_noise_ratio_is_one_for_unchanged_or_uniformly_scaled_noise(
    rng: np.random.Generator,
) -> None:
    clean, noisy = _noise_floor_scene(rng)
    assert musical_noise_ratio(clean, noisy, noisy) == pytest.approx(1.0)
    assert musical_noise_ratio(clean, noisy, 0.3 * noisy) == pytest.approx(1.0)


def test_musical_noise_ratio_rises_when_only_isolated_cells_survive(
    rng: np.random.Generator,
) -> None:
    clean, noisy = _noise_floor_scene(rng)
    spectrum = stft(noisy)
    mask = np.where(rng.random(spectrum.shape) < 0.05, 1.0, 0.05)
    spiky = istft(spectrum * mask, noisy.size)
    smooth = istft(spectrum * 0.2, noisy.size)
    assert musical_noise_ratio(clean, noisy, spiky) > 3.0
    assert musical_noise_ratio(clean, noisy, smooth) == pytest.approx(1.0)


def test_musical_noise_ratio_needs_enough_noise_only_frames(rng: np.random.Generator) -> None:
    x = rng.standard_normal(4000)
    assert math.isnan(musical_noise_ratio(x, x, x))


# ------------------------------------------------------------ external metrics (fake and real)
def _install_fakes(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, object]]) -> None:
    class NoUtterancesError(Exception):
        pass

    fake_pesq = types.ModuleType("pesq")

    def pesq(fs: int, ref: np.ndarray, deg: np.ndarray, mode: str) -> float:
        calls.append(("pesq", (fs, mode)))
        if np.max(np.abs(deg)) < 1e-3:
            raise NoUtterancesError("none")
        return 3.5

    fake_pesq.pesq = pesq  # type: ignore[attr-defined]
    fake_stoi = types.ModuleType("pystoi")

    def stoi(ref: np.ndarray, deg: np.ndarray, fs: int, extended: bool = False) -> float:
        calls.append(("stoi", (fs, extended)))
        return 0.9

    fake_stoi.stoi = stoi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pesq", fake_pesq)
    monkeypatch.setitem(sys.modules, "pystoi", fake_stoi)


def test_pesq_wrapper_passes_wide_band_mode_and_handles_edge_cases(
    monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator
) -> None:
    calls: list[tuple[str, object]] = []
    _install_fakes(monkeypatch, calls)
    x = rng.standard_normal(FS)
    assert pesq_wb(x, x, FS) == 3.5
    assert calls == [("pesq", (16000, "wb"))]
    assert math.isnan(pesq_wb(x, np.zeros(FS), FS))
    assert math.isnan(pesq_wb(np.zeros(FS), x, FS))
    assert len(calls) == 1  # silent inputs never reach the package
    assert math.isnan(pesq_wb(x, 1e-6 * x, FS))  # package reports "no utterance"
    with pytest.raises(ValueError, match="16000"):
        pesq_wb(x, x, 8000)
    with pytest.raises(ValueError, match="quarter|at least"):
        pesq_wb(x[:1000], x[:1000], FS)
    with pytest.raises(ValueError, match="same length"):
        pesq_wb(x, x[:-1], FS)


def test_pesq_wrapper_does_not_swallow_other_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = types.ModuleType("pesq")

    def pesq(*args: object) -> float:
        raise RuntimeError("boom")

    broken.pesq = pesq  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pesq", broken)
    x = np.ones(FS)
    with pytest.raises(RuntimeError, match="boom"):
        pesq_wb(x, x, FS)


def test_stoi_wrapper_converts_the_too_few_frames_warning_to_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("pystoi")

    def stoi(ref: np.ndarray, deg: np.ndarray, fs: int, extended: bool = False) -> float:
        warnings.warn(
            "Not enough STFT frames to compute intermediate intelligibility", stacklevel=2
        )
        return 1e-5

    fake.stoi = stoi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pystoi", fake)
    assert math.isnan(stoi_score(np.ones(100), np.ones(100), FS))


def test_stoi_wrapper_passes_through_a_normal_value(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    _install_fakes(monkeypatch, calls)
    x = np.ones(2000)
    assert stoi_score(x, x, FS) == 0.9
    assert stoi_score(x, x, FS, extended=True) == 0.9
    assert calls == [("stoi", (16000, False)), ("stoi", (16000, True))]
    with pytest.raises(ValueError, match="same length"):
        stoi_score(x, x[:-1], FS)


@pytest.mark.parametrize("package", ["pesq", "pystoi"])
def test_missing_external_package_gives_a_clear_error(
    package: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, package, None)
    x = np.ones(FS)
    with pytest.raises(ImportError, match="pip install"):
        (pesq_wb if package == "pesq" else stoi_score)(x, x, FS)


def test_external_versions_report_missing_packages_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert external_versions() == {"pesq": "", "pystoi": ""}


def test_real_pesq_and_stoi_behave_sensibly(rng: np.random.Generator) -> None:
    pytest.importorskip("pesq")
    pytest.importorskip("pystoi")
    t = np.arange(3 * FS) / FS
    voiced = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 12))
    x = 0.2 * voiced * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    noisy = x + 0.05 * rng.standard_normal(x.size)
    assert pesq_wb(x, x, FS) > 4.4
    assert pesq_wb(x, noisy, FS) < pesq_wb(x, x, FS)
    assert stoi_score(x, x, FS) > 0.99
    assert stoi_score(x, noisy, FS) < stoi_score(x, x, FS)
    assert math.isnan(pesq_wb(x, np.zeros_like(x), FS))
    versions = external_versions()
    assert versions["pesq"] and versions["pystoi"]


# ------------------------------------------------------------ evaluate()
def _scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    clean, _ = _noise_floor_scene(rng)
    return clean, mix_at_snr(clean, rng.standard_normal(clean.size), 5.0).noisy


def test_an_unprocessed_signal_scores_zero_improvement(rng: np.random.Generator) -> None:
    clean, noisy = _scene(rng)
    result = evaluate(clean, noisy, noisy, external=False)
    assert result.delta_snr_db == pytest.approx(0.0)
    assert result.delta_segsnr_db == pytest.approx(0.0)
    assert result.noise_reduction_db == pytest.approx(0.0)
    assert result.musical_noise_ratio == pytest.approx(1.0)
    assert result.lsd_in_db == result.lsd_out_db
    assert result.snr_in_db == pytest.approx(5.0, abs=1e-9)
    assert math.isnan(result.speech_distortion_db)
    assert all(math.isnan(v) for v in (result.pesq_in, result.pesq_out, result.stoi_in))


def test_evaluate_fills_everything_when_components_and_externals_are_given(
    monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator
) -> None:
    _install_fakes(monkeypatch, [])
    clean, noisy = _scene(rng)
    enhanced = clean + 0.5 * (noisy - clean)
    result = evaluate(clean, noisy, enhanced, speech_component=clean, external=True)
    assert result.delta_snr_db == pytest.approx(10.0 * math.log10(4.0), abs=1e-9)
    assert result.pesq_out == 3.5
    assert result.stoi_in == 0.9
    assert result.speech_distortion_db < -200.0
    assert 0.0 < result.active_fraction < 1.0
    assert not any(math.isnan(v) for v in result.as_dict().values())


def test_metrics_dict_has_one_entry_per_field(rng: np.random.Generator) -> None:
    clean, noisy = _scene(rng)
    result = evaluate(clean, noisy, noisy, external=False)
    assert isinstance(result, Metrics)
    assert set(result.as_dict()) == set(Metrics.__dataclass_fields__)


def test_evaluate_rejects_mismatched_lengths(rng: np.random.Generator) -> None:
    clean, noisy = _scene(rng)
    with pytest.raises(ValueError, match="same length"):
        evaluate(clean, noisy, noisy[:-1], external=False)
