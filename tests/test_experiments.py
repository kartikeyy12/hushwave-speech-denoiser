from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hushwave import tuning
from hushwave.corpus import (
    Case,
    NoiseItem,
    SpeechItem,
    build_cases,
    clear_audio_cache,
)
from hushwave.experiments import (
    evaluate_case,
    print_progress,
    run_jobs,
    tune_case,
)
from hushwave.systems import System, b5_from

FS = 16_000


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = round(seconds * FS)
    t = np.arange(n) / FS
    carrier = sum(
        np.sin(2 * np.pi * f * t * (1 + 0.01 * rng.random())) for f in (150, 300, 900, 2400)
    )
    envelope = np.clip(np.sin(2 * np.pi * 1.7 * t), 0, None) ** 2
    return 0.1 * carrier * envelope + 1e-4 * rng.standard_normal(n)


@pytest.fixture
def tiny_cases(tmp_path: Path) -> list[Case]:
    clear_audio_cache()
    speech = []
    for i in range(3):
        path = tmp_path / f"u{i}.wav"
        sf.write(str(path), _speechlike(3.2, i), FS, subtype="PCM_16")
        speech.append(SpeechItem("libri", f"u{i}", f"s{i}", str(path)))
    noises = [NoiseItem("synthetic", "syn-white"), NoiseItem("synthetic", "syn-pink")]
    return build_cases("dev", speech, noises, (0.0, 10.0))


def _systems() -> list[System]:
    b5 = b5_from(System("x", "x", "gain", prefilter="spectral").with_est(rise_db_per_s=3.0))
    return [
        System("B0", "raw", "unprocessed", external=True),
        System("B2", "w", "scipy_wiener", window=9, external=True),
        b5,
    ]


def test_evaluate_case_returns_complete_rows(tiny_cases: list[Case]) -> None:
    rows = evaluate_case(tiny_cases[0], _systems())
    assert [r["system"] for r in rows] == ["B0", "B2", "B5"]
    b0, _, b5 = rows
    assert b0["delta_snr_db"] == pytest.approx(0.0)
    assert b0["delta_lsd_db"] == 0.0
    assert math.isnan(b0["speech_distortion_db"])
    assert not math.isnan(b5["speech_distortion_db"])
    for row in rows:
        assert row["case"] == tiny_cases[0].case_id and row["snr_in"] == 0.0
        assert "pesq_out" in row and "delta_stoi" in row and "external_note" in row
    assert b5["noise_kind"] == "synthetic" and b5["corpus"] == "libri"


def test_external_failures_are_recorded_not_raised(
    tiny_cases: list[Case], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> float:
        raise RuntimeError("pesq exploded")

    monkeypatch.setattr("hushwave.experiments.pesq_wb", boom)
    rows = evaluate_case(tiny_cases[0], _systems())
    assert math.isnan(rows[2]["pesq_out"])
    assert "pesq exploded" in rows[2]["external_note"]


def test_tune_case_is_cheap_and_consistent_with_full(tiny_cases: list[Case]) -> None:
    systems = _systems()
    cheap = tune_case(tiny_cases[1], systems)
    full = evaluate_case(tiny_cases[1], systems)
    for c, f in zip(cheap, full, strict=True):
        assert c["delta_snr_db"] == pytest.approx(f["delta_snr_db"], abs=1e-9)
        assert c["delta_segsnr_db"] == pytest.approx(f["delta_segsnr_db"], abs=1e-9)


def test_run_jobs_is_identical_for_any_worker_count(tiny_cases: list[Case]) -> None:
    systems = _systems()[:1] + _systems()[2:]
    seen: list[tuple[int, int]] = []
    one = run_jobs(tiny_cases[:4], systems, "tune", 1, lambda d, t, e: seen.append((d, t)))
    two = run_jobs(tiny_cases[:4], systems, "tune", 2)
    assert one == two
    assert seen[-1] == (4, 4)
    with pytest.raises(ValueError):
        run_jobs(tiny_cases, systems, "other")
    with pytest.raises(ValueError):
        run_jobs(tiny_cases, systems, "tune", 0)


def test_print_progress_reports_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    report = print_progress("unit")
    report(1, 4, 2.0)
    report(2, 4, 4.0)  # throttled
    report(4, 4, 8.0)
    err = capsys.readouterr().err
    assert "[unit] 1/4" in err and "4/4" in err and "2/4" not in err


def test_objective_ignores_non_finite_rows() -> None:
    rows = [
        {"system": "a", "delta_snr_db": 2.0, "delta_segsnr_db": 4.0},
        {"system": "a", "delta_snr_db": math.nan, "delta_segsnr_db": 9.0},
        {"system": "a", "delta_snr_db": 4.0, "delta_segsnr_db": 6.0},
        {"system": "b", "delta_snr_db": -1.0, "delta_segsnr_db": -1.0},
    ]
    table = tuning.objective_by_system(rows)
    assert table["a"] == pytest.approx((4.0, 3.0, 5.0))
    assert table["b"][0] == -1.0


def test_best_prefers_earlier_on_ties() -> None:
    assert tuning._best(["a", "b", "c"], {"a": 1.0, "b": 1.0 + 1e-12, "c": 0.5}) == "a"
    assert tuning._best(["a", "b"], {"a": 1.0, "b": 1.5}) == "b"
    assert tuning._best(["a", "b"], {"b": 1.0}) == "b"


def test_run_tuning_end_to_end_with_small_grids(
    tiny_cases: list[Case], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tuning, "RULES", ("lsa",))
    monkeypatch.setattr(tuning, "ESTIMATORS", ("mcra", "minstats"))
    monkeypatch.setattr(tuning, "PREFILTERS", ("none", "spectral"))
    monkeypatch.setattr(tuning, "GAIN_GRID", {"g_min": (0.1, 0.3), "gain_mu": (0.5, 0.0)})
    monkeypatch.setattr(tuning, "RISE_GRID", (3.0, 6.0))
    monkeypatch.setattr(tuning, "MCRA_DELTA_GRID", (5.0, 3.0))
    monkeypatch.setattr(tuning, "MINSTATS_ALPHA_GRID", (0.9, 0.7))
    monkeypatch.setattr(tuning, "B1_GRID", {"ss_a": (2.0, 1.0), "g_min": (0.1,)})
    monkeypatch.setattr(tuning, "B4_GRID", (0.1, 0.2))
    monkeypatch.setattr(tuning, "B2_WINDOWS", (15, 5))
    result = tuning.run_tuning(tiny_cases[:4])
    assert result.b5.name == "B5" and result.b5.external
    assert result.n_cases == 4
    assert result.strongest_baseline in {"B1", "B2", "B3", "B4"}
    assert set(result.scores) == {"B1", "B2", "B3", "B4", "B5"}
    stages = {entry["stage"] for entry in result.log}
    assert {"structure", "gain-grid", "sweep", "baselines"} <= stages
    grid = [e for e in result.log if e["stage"] == "gain-grid"]
    assert len(grid) == 4  # full factorial of the two monkeypatched gain parameters
    assert {(e["gain"]["g_min"], e["gain"]["gain_mu"]) for e in grid} == {
        (0.1, 0.5), (0.1, 0.0), (0.3, 0.5), (0.3, 0.0),
    }  # fmt: skip
    assert result.scores["B5"] >= max(e["J"] for e in grid) - 1e-9
    structure_scores = [e["J"] for e in result.log if e["stage"] == "structure"]
    assert result.scores["B5"] >= max(structure_scores) - 1e-9
    assert dict(result.b1_gain)["gain_mu"] == 0.0 and result.b2_window in (15, 5)
    with pytest.raises(ValueError):
        tuning.run_tuning([])
