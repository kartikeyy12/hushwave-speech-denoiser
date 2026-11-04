from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hushwave.nonintrusive import (
    block_energy_db,
    evaluate_recording,
    nonintrusive_metrics,
    pause_psd,
    segment_by_noise_floor,
)
from hushwave.systems import System, b5_from

FS = 16_000


def _scene(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A fan-like noise reference and the same noise with tone bursts (speech stand-in) on top."""
    rng = np.random.default_rng(seed)
    n = 12 * FS
    noise = 0.01 * rng.standard_normal(n)
    reference = 0.01 * np.random.default_rng(seed + 1).standard_normal(n)
    t = np.arange(n) / FS
    burst = np.zeros(n)
    for start in (1.0, 4.0, 7.0, 10.0):
        a, b = round(start * FS), round((start + 1.5) * FS)
        burst[a:b] = 0.3 * np.sin(2 * np.pi * 300 * t[a:b]) * np.hanning(b - a)
    return noise + burst, reference


def test_block_energy_and_segmentation() -> None:
    noisy, reference = _scene()
    with pytest.raises(ValueError):
        block_energy_db(np.zeros(10), 256)
    seg = segment_by_noise_floor(noisy, reference)
    assert seg.pause.size == noisy.size // 256
    assert not np.any(seg.pause & seg.speech)
    assert 0.4 < seg.pause.mean() < 0.9 and 0.1 < seg.speech.mean() < 0.5
    first_burst = slice(round(1.5 * FS / 256), round(2.2 * FS / 256))
    assert seg.speech[first_burst].all()
    rough = segment_by_noise_floor(noisy, None)
    assert rough.floor_db != seg.floor_db and rough.pause.any()


def test_pause_psd_matches_the_noise_level() -> None:
    noisy, reference = _scene()
    seg = segment_by_noise_floor(noisy, reference)
    freqs, psd = pause_psd(noisy, seg.pause)  # type: ignore[misc]
    expected = 0.01**2 / (FS / 2)
    assert freqs[-1] == FS / 2
    assert np.median(psd) == pytest.approx(expected, rel=0.15)
    assert pause_psd(noisy, np.zeros_like(seg.pause)) is None


def test_metrics_on_a_known_attenuation() -> None:
    noisy, reference = _scene()
    seg = segment_by_noise_floor(noisy, reference)
    enhanced = noisy.copy()
    mask = np.repeat(seg.pause, 256)
    enhanced[: mask.size][mask] *= 0.1
    metrics = nonintrusive_metrics(noisy, enhanced, seg)
    assert metrics["noise_reduction_pause_db"] == pytest.approx(20.0, abs=0.01)
    assert metrics["speech_level_change_db"] == pytest.approx(0.0, abs=0.5)
    assert metrics["residual_pause_dbfs"] < -50.0
    assert 0.5 < metrics["musical_noise_ratio"] < 2.0
    none = segment_by_noise_floor(np.full(8 * FS, 0.5), None)
    quiet = nonintrusive_metrics(np.full(8 * FS, 0.5), np.zeros(8 * FS), none)
    assert math.isnan(quiet["speech_level_change_db"]) or math.isfinite(
        quiet["speech_level_change_db"]
    )


def test_evaluate_recording_end_to_end(tmp_path: Path) -> None:
    noisy, reference = _scene()
    b5 = b5_from(System("x", "x", "gain", prefilter="spectral").with_est(rise_db_per_s=3.0))
    systems = [
        System("B0", "raw", "unprocessed"),
        System("B1", "s", "gain", rule="subtraction"),
        b5,
    ]
    result = evaluate_recording(noisy, reference, systems, save_dir=tmp_path, stem="scene")
    assert result["floor_from_noise_recording"] is True
    assert result["pause_blocks"] + result["speech_blocks"] <= result["total_blocks"]
    assert set(result["systems"]) == {"B0", "B1", "B5"}
    assert result["systems"]["B0"]["noise_reduction_pause_db"] == pytest.approx(0.0, abs=1e-9)
    assert result["systems"]["B5"]["noise_reduction_pause_db"] > 3.0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scene_B1.wav", "scene_B5.wav"]
    assert sf.info(str(tmp_path / "scene_B5.wav")).frames == noisy.size
    psd = result["psd"]
    assert {"freq_hz", "B0_db", "B1_db", "B5_db", "noise_reference_db"} <= set(psd)
    assert float(np.median(psd["B5_db"])) < float(np.median(psd["B0_db"]))
    no_reference = evaluate_recording(noisy, None, systems[:1])
    assert no_reference["floor_from_noise_recording"] is False
    assert "noise_reference_db" not in no_reference["psd"]


def test_signal_length_need_not_be_a_multiple_of_the_hop() -> None:
    noisy, reference = _scene()
    odd = noisy[: noisy.size - 100]
    seg = segment_by_noise_floor(odd, reference)
    assert seg.pause.size == -(-odd.size // 256) and not seg.pause[-1] and not seg.speech[-1]
    metrics = nonintrusive_metrics(odd, 0.5 * odd, seg)
    assert metrics["noise_reduction_pause_db"] == pytest.approx(6.02, abs=0.01)
    from hushwave.metrics import musical_noise_masked, noise_reduction_masked

    with pytest.raises(ValueError, match="one flag per block"):
        noise_reduction_masked(odd, odd, seg.pause[:-1])
    with pytest.raises(ValueError, match="one flag per block"):
        musical_noise_masked(odd, odd, seg.pause[:-1])
