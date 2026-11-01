"""Batch enhance() and StreamingEnhancer: equality, fixed delay, state, edge cases."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hushwave.baselines import spectral_denoise
from hushwave.config import DEFAULT, Config
from hushwave.enhancer import (
    StreamingEnhancer,
    enhance,
    measure_delay,
    measure_memory,
    measure_timing,
    performance_report,
    stream,
)
from hushwave.filters import frequency_response, highpass_biquad
from hushwave.gains import RULES

FS = DEFAULT.fs
N = DEFAULT.frame_len
H = DEFAULT.hop
D = N - 1
# The two implementations perform the same arithmetic frame by frame; measured differences are
# at the 1e-16 level. The acceptance threshold from the specification is 1e-9.
TOL = 1e-9


def noise(rng: np.random.Generator, n: int, scale: float = 0.1) -> np.ndarray:
    return scale * rng.standard_normal(n)


# ------------------------------------------------------------ streaming == batch
@pytest.mark.parametrize("length", [1, 2, 255, 256, 257, 511, 512, 513, 1000, 4097])
@pytest.mark.parametrize("blocks", [[1], [7], [255], [256], [257], [512], [1000, 3, 50], [9000]])
def test_streaming_equals_batch(length: int, blocks: list[int], rng: np.random.Generator) -> None:
    x = noise(rng, length)
    batch = enhance(x)
    streamed = stream(x, blocks)
    assert streamed.shape == batch.shape
    assert np.max(np.abs(streamed - batch)) < TOL


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("estimator", ["mcra", "minstats"])
@pytest.mark.parametrize("prefilter", [True, False])
def test_every_configuration_streams_like_it_batches(
    rule: str, estimator: str, prefilter: bool, rng: np.random.Generator
) -> None:
    x = noise(rng, 3000)
    batch = enhance(x, rule=rule, estimator=estimator, prefilter=prefilter)
    streamed = stream(x, [123, 480, 1], rule=rule, estimator=estimator, prefilter=prefilter)
    assert np.max(np.abs(streamed - batch)) < TOL


def test_overrides_stream_like_they_batch(rng: np.random.Generator) -> None:
    x = noise(rng, 3000)
    gain = {"g_min": 0.3, "gain_mu": 0.7, "dd_alpha": 0.9}
    tracker = {"alpha_s": 0.7}
    batch = enhance(x, gain_overrides=gain, estimator_overrides=tracker)
    streamed = stream(x, [200], gain_overrides=gain, estimator_overrides=tracker)
    assert np.max(np.abs(streamed - batch)) < TOL


@pytest.mark.parametrize(("frame_len", "hop"), [(256, 128), (1024, 512)])
def test_other_frame_lengths_stream_like_they_batch(
    frame_len: int, hop: int, rng: np.random.Generator
) -> None:
    cfg = Config(frame_len=frame_len, hop=hop)
    x = noise(rng, 6000)
    assert np.max(np.abs(stream(x, [333, 17], cfg) - enhance(x, cfg))) < TOL


@given(
    length=st.integers(min_value=1, max_value=3000),
    blocks=st.lists(st.integers(min_value=1, max_value=1500), min_size=1, max_size=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=40, deadline=None)
def test_random_block_sequences_stream_like_batch(
    length: int, blocks: list[int], seed: int
) -> None:
    x = np.random.default_rng(seed).standard_normal(length) * 0.1
    assert np.max(np.abs(stream(x, blocks) - enhance(x))) < TOL


def test_enhance_without_prefilter_is_the_shared_offline_pipeline(
    rng: np.random.Generator,
) -> None:
    x = noise(rng, 4000)
    np.testing.assert_array_equal(
        enhance(x, rule="wiener", prefilter=False), spectral_denoise(x, "wiener")
    )


# ------------------------------------------------------------ fixed delay and block sizes
def test_every_call_returns_as_many_samples_as_it_was_given(rng: np.random.Generator) -> None:
    engine = StreamingEnhancer()
    x = noise(rng, 5000)
    pos = 0
    for size in (1, 2, 3, 100, 255, 256, 257, 700, 1, 1000, 2000):
        block = x[pos : pos + size]
        assert engine.process(block).shape == block.shape
        pos += block.size


def test_stream_starts_with_delay_zeros_and_flush_returns_the_last_delay_samples(
    rng: np.random.Generator,
) -> None:
    x = noise(rng, 3000)
    engine = StreamingEnhancer()
    first = engine.process(x[:700])
    assert np.all(first[:D] == 0.0)
    rest = engine.process(x[700:])
    tail = engine.flush()
    assert tail.shape == (D,)
    full = np.concatenate((first, rest, tail))
    assert full.shape == (x.size + D,)
    assert np.max(np.abs(full[D:] - enhance(x))) < TOL


@pytest.mark.parametrize(
    "config", [DEFAULT, Config(frame_len=256, hop=128), Config(frame_len=1024, hop=512)]
)
def test_measured_delay_is_frame_len_minus_one(config: Config) -> None:
    result = measure_delay(config)
    assert result.expected_samples == config.frame_len - 1
    assert result.measured_samples == result.expected_samples


def test_delay_properties_match_the_config() -> None:
    engine = StreamingEnhancer()
    assert engine.delay_samples == DEFAULT.latency_max_samples == 511
    assert engine.delay_ms == pytest.approx(1000.0 * 511 / FS)


def test_unit_gain_without_prefilter_is_a_pure_delay(rng: np.random.Generator) -> None:
    x = noise(rng, 3000)
    out = stream(x, [111], prefilter=False, gain_overrides={"g_min": 1.0})
    np.testing.assert_allclose(out, x, atol=1e-10)


def test_prefilter_follows_the_designed_high_pass(rng: np.random.Generator) -> None:
    b, a = highpass_biquad(DEFAULT.hp_cutoff_hz, FS)
    for freq in (20.0, 80.0, 400.0):
        t = np.arange(4 * FS) / FS
        tone = 0.2 * np.sin(2.0 * np.pi * freq * t)
        out = enhance(tone, gain_overrides={"g_min": 1.0})
        measured = np.sqrt(np.mean(out[2 * FS :] ** 2) / np.mean(tone[2 * FS :] ** 2))
        expected = float(np.abs(frequency_response(b, a, [freq], FS)[0]))
        assert measured == pytest.approx(expected, rel=0.02)


# ------------------------------------------------------------ state, reset, memory
def test_flush_resets_so_the_next_stream_starts_fresh(rng: np.random.Generator) -> None:
    x = noise(rng, 2500)
    engine = StreamingEnhancer()
    first = np.concatenate([engine.process(x), engine.flush()])
    second = np.concatenate([engine.process(x), engine.flush()])
    np.testing.assert_array_equal(first, second)


def test_reset_restores_the_initial_state(rng: np.random.Generator) -> None:
    x = noise(rng, 2500)
    engine = StreamingEnhancer()
    reference = np.concatenate([engine.process(x[:900]), engine.process(x[900:])])
    engine.process(noise(rng, 3333, 2.0))
    engine.reset()
    again = np.concatenate([engine.process(x[:900]), engine.process(x[900:])])
    np.testing.assert_array_equal(reference, again)


def test_engine_buffers_do_not_grow_with_stream_length(rng: np.random.Generator) -> None:
    engine = StreamingEnhancer()
    bound = 8 * (N + H + (D - H))
    for _ in range(200):
        engine.process(noise(rng, 333))
        assert engine.state_bytes() <= bound


def test_peak_memory_is_independent_of_duration() -> None:
    short = measure_memory(seconds=3.0)
    long = measure_memory(seconds=12.0)
    assert long < 1.25 * short


# ------------------------------------------------------------ validation and edge cases
def test_empty_block_and_empty_flush() -> None:
    engine = StreamingEnhancer()
    assert engine.process(np.zeros(0)).size == 0
    assert engine.flush().size == 0


@pytest.mark.parametrize("bad", [np.ones((2, 2)), np.array([1.0, np.nan]), np.array([np.inf])])
def test_bad_blocks_are_rejected_without_touching_the_state(
    bad: np.ndarray, rng: np.random.Generator
) -> None:
    x = noise(rng, 2000)
    clean = StreamingEnhancer()
    expected = np.concatenate([clean.process(x), clean.flush()])
    engine = StreamingEnhancer()
    first = engine.process(x[:700])
    with pytest.raises(ValueError):
        engine.process(bad)
    got = np.concatenate([first, engine.process(x[700:]), engine.flush()])
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("bad", [np.ones((2, 2)), np.array([]), np.array([np.nan, 1.0])])
def test_enhance_rejects_bad_signals(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        enhance(bad)


@pytest.mark.parametrize("sizes", [[], [0], [3, -1]])
def test_stream_rejects_bad_block_sizes(sizes: list[int]) -> None:
    with pytest.raises(ValueError, match="block_sizes"):
        stream(np.ones(100), sizes)


def test_silence_stays_exactly_silent() -> None:
    x = np.zeros(3000)
    assert np.all(enhance(x) == 0.0)
    assert np.all(stream(x, [100]) == 0.0)


def test_dc_input_is_finite_and_the_high_pass_removes_it() -> None:
    out = enhance(np.full(3 * FS, 0.3))
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out[2 * FS :])) < 1e-3


def test_clipped_input_gives_finite_output() -> None:
    t = np.arange(FS) / FS
    square = np.sign(np.sin(2.0 * np.pi * 220.0 * t))
    out = enhance(square)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) < 2.0


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e6])
def test_extreme_levels_never_produce_nan(scale: float, rng: np.random.Generator) -> None:
    out = enhance(scale * rng.standard_normal(4000))
    assert np.all(np.isfinite(out))


def test_very_long_input_streams_correctly(rng: np.random.Generator) -> None:
    x = noise(rng, 30 * FS, 0.05)
    assert np.max(np.abs(stream(x, [256]) - enhance(x))) < TOL


def test_noise_only_input_is_reduced() -> None:
    x = 0.05 * np.random.default_rng(11).standard_normal(5 * FS)
    out = enhance(x)
    change = 10.0 * np.log10(np.mean(out[FS:] ** 2) / np.mean(x[FS:] ** 2))
    assert change < -10.0


# ------------------------------------------------------------ measurements
def test_timing_measurement_reports_sensible_numbers() -> None:
    (timing,) = measure_timing(seconds=1.0, blocks=(256,))
    assert timing.block == 256
    assert timing.block_budget_ms == pytest.approx(16.0)
    assert 0.0 < timing.seconds_per_audio_second < 1.0
    assert timing.worst_block_ms > 0.0


def test_performance_report_is_json_ready_and_consistent() -> None:
    import json

    report = performance_report(seconds=1.0)
    json.dumps(report)
    assert report["algorithmic_delay_samples"] == report["measured_delay_samples"] == D
    assert report["per_sample_delay_range_samples"] == [DEFAULT.latency_min_samples, D]
    assert report["streaming_vs_batch_max_abs_difference"] < TOL
    assert len(report["timing"]) == 4
