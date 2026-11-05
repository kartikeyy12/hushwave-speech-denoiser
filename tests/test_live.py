"""Frozen B5 on the streaming engine, and the block-by-block session wrapper (M9)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hushwave.live import (
    LEVEL_FLOOR_DB,
    MAX_BLOCK,
    LiveSession,
    build_engine,
    level_dbfs,
    load_b1,
    load_b5,
    load_tuning,
    render,
)
from hushwave.systems import System


def _noise(seconds: float, rng: np.random.Generator, scale: float = 0.05) -> np.ndarray:
    return scale * rng.standard_normal(round(16_000 * seconds))


def test_frozen_b5_matches_the_tuning_file() -> None:
    b5 = load_b5()
    final = load_tuning()["final"]
    assert isinstance(final, dict)
    assert b5.name == "B5"
    assert b5.rule == final["b5"]["rule"]
    assert b5.estimator == final["b5"]["estimator"]
    assert b5.prefilter == final["b5"]["prefilter"]
    assert dict(b5.gain) == {k: float(v) for k, v in final["b5"]["gain"]}


def test_b1_is_spectral_subtraction_on_the_same_pipeline() -> None:
    b1, b5 = load_b1(), load_b5()
    assert b1.name == "B1"
    assert b1.rule == "subtraction"
    assert (b1.estimator, b1.prefilter, b1.est) == (b5.estimator, b5.prefilter, b5.est)


def test_missing_tuning_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tuning stage"):
        load_tuning(tmp_path / "nope.json")


def test_malformed_tuning_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps({"final": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="final"):
        load_b5(path)
    with pytest.raises(ValueError, match="final"):
        load_b1(path)
    path.write_text(
        json.dumps({"final": {"b5": {"name": "B5", "label": "x", "kind": "gain"}, "b1_gain": 3}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pairs"):
        load_b1(path)


def test_streaming_equals_batch_for_the_frozen_system(rng: np.random.Generator) -> None:
    system = load_b5()
    x = _noise(2.5, rng)
    engine = build_engine(system)
    sizes = [256, 1, 777, 256, 4096, 3]
    pieces, start, i = [], 0, 0
    while start < x.size:
        size = sizes[i % len(sizes)]
        pieces.append(engine.process(x[start : start + size]))
        start += size
        i += 1
    streamed = np.concatenate(pieces)
    batch = render(system, x)
    delay = engine.delay_samples
    assert np.max(np.abs(streamed[delay:] - batch[: streamed.size - delay])) < 1e-9


def test_only_gain_systems_run_on_the_engine() -> None:
    with pytest.raises(ValueError, match="gain-based"):
        build_engine(System("B0", "unprocessed", "unprocessed"))


def test_session_reports_the_engine_delay() -> None:
    session = LiveSession(load_b5())
    assert session.fs == 16_000
    assert session.delay_samples == 511
    assert session.delay_ms == pytest.approx(31.9375)


def test_session_output_length_and_stats(rng: np.random.Generator) -> None:
    session = LiveSession(load_b5())
    block = _noise(0.016, rng)
    out, stats = session.process(block)
    assert out.shape == block.shape
    assert stats.compute_ms >= 0.0
    assert stats.level_in_dbfs == pytest.approx(level_dbfs(block))
    assert session.blocks == 1 and session.samples == block.size


def test_session_bypass_passes_the_input_after_the_fade(rng: np.random.Generator) -> None:
    session = LiveSession(load_b5())
    x = _noise(1.0, rng)
    blocks = [x[i : i + 256] for i in range(0, 12 * 256, 256)]
    for block in blocks[:6]:
        session.process(block, True)
    fade, _ = session.process(blocks[6], False)
    assert not np.allclose(fade, blocks[6])
    for block in blocks[7:]:
        out, _ = session.process(block, False)
        np.testing.assert_allclose(out, block, atol=1e-12)


def test_session_fade_is_continuous_at_the_edges(rng: np.random.Generator) -> None:
    session = LiveSession(load_b5())
    block = _noise(0.016, rng)
    for _ in range(4):
        session.process(block, True)
    wet_only = LiveSession(load_b5())
    for _ in range(4):
        wet_only.process(block, True)
    wet, _ = wet_only.process(block, True)
    faded, _ = session.process(block, False)
    n = block.size
    assert faded[0] == pytest.approx(wet[0] * (1 - 1 / n) + block[0] / n)
    assert faded[-1] == pytest.approx(block[-1])


def test_session_keeps_the_engine_running_while_bypassed(rng: np.random.Generator) -> None:
    a, b = LiveSession(load_b5()), LiveSession(load_b5())
    x = _noise(1.5, rng)
    blocks = [x[i : i + 256] for i in range(0, x.size - 255, 256)]
    for block in blocks[:20]:
        a.process(block, False)
        b.process(block, True)
    out_a, _ = a.process(blocks[20], True)
    out_b, _ = b.process(blocks[20], True)
    wet_a = a.engine
    assert wet_a.delay_samples == b.engine.delay_samples
    assert np.max(np.abs(out_a - out_b)) <= 1.0  # fade blends the dry block in
    out_a2, _ = a.process(blocks[21], True)
    out_b2, _ = b.process(blocks[21], True)
    np.testing.assert_allclose(out_a2, out_b2, atol=1e-12)


@pytest.mark.parametrize("bad", [np.zeros(0), np.zeros((2, 2)), np.zeros(MAX_BLOCK + 1)])
def test_session_rejects_bad_shapes(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="block must be"):
        LiveSession(load_b5()).process(bad)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_session_rejects_non_finite_input(value: float) -> None:
    block = np.zeros(256)
    block[3] = value
    with pytest.raises(ValueError, match="NaN or Inf"):
        LiveSession(load_b5()).process(block)


def test_output_is_clipped_to_full_scale() -> None:
    session = LiveSession(load_b5())
    out, _ = session.process(np.full(256, 5.0), False)
    assert np.max(out) == 1.0


def test_level_dbfs() -> None:
    assert level_dbfs(np.zeros(10)) == LEVEL_FLOOR_DB
    assert level_dbfs(np.zeros(0)) == LEVEL_FLOOR_DB
    assert level_dbfs(np.ones(10)) == pytest.approx(0.0)
    assert level_dbfs(0.1 * np.ones(10)) == pytest.approx(-20.0)
    assert level_dbfs(1e-9 * np.ones(10)) == LEVEL_FLOOR_DB
