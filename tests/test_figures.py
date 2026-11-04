from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fakes import fake_results
from hushwave import figures
from hushwave.corpus import Realized, clear_audio_cache
from hushwave.pipeline import Sources
from hushwave.systems import System, b5_from

FS = 16_000


def test_table_plots_write_files(tmp_path: Path) -> None:
    results = fake_results()
    for function, name in (
        (figures.plot_delta_vs_snr, "a.png"),
        (figures.plot_ablation, "b.png"),
        (figures.plot_by_noise, "c.png"),
        (figures.plot_pause_psd, "d.png"),
    ):
        path = function(results, tmp_path / name)
        assert path is not None and path.stat().st_size > 5_000
    assert figures.plot_pause_psd({"real_recordings": {}}, tmp_path / "none.png") is None


def test_latency_and_tracking_plots(tmp_path: Path) -> None:
    report = {
        "algorithmic_delay_ms": 48.0,
        "timing": [
            {"block": 64, "real_time_factor": 0.2},
            {"block": 512, "real_time_factor": 0.05},
        ],
    }
    assert figures.plot_latency(report, tmp_path / "l.png").stat().st_size > 5_000
    assert figures.plot_noise_tracking(tmp_path / "t.png").stat().st_size > 5_000


def _data() -> Realized:
    rng = np.random.default_rng(1)
    t = np.arange(3 * FS) / FS
    clean = 0.2 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)
    noise = 0.05 * rng.standard_normal(clean.size)
    return Realized(clean, noise, clean + noise)


def test_example_plots(tmp_path: Path) -> None:
    system = b5_from(System("x", "x", "gain", prefilter="spectral"))
    paths = figures.plot_example(_data(), system, "example", tmp_path)
    assert [p.name for p in paths] == [
        "fig_example_waveforms.png",
        "fig_example_spectrograms.png",
        "fig_gain_per_band.png",
    ]
    assert all(p.stat().st_size > 5_000 for p in paths)


def test_make_all_with_and_without_data(tmp_path: Path) -> None:
    clear_audio_cache()
    rng = np.random.default_rng(2)
    rec = tmp_path / "Speech_quiet.wav"
    t = np.arange(40 * FS) / FS
    speech = 0.2 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 0.4 * t) > -0.2)
    sf.write(str(rec), speech + 1e-4 * rng.standard_normal(t.size), FS, subtype="PCM_16")
    fan = tmp_path / "Fan_noise.wav"
    sf.write(str(fan), 0.03 * rng.standard_normal(40 * FS), FS, subtype="PCM_16")
    sources = Sources(tmp_path, None, {"speech_quiet": rec, "fan_noise": fan})
    case_id = ""
    from hushwave.corpus import build_cases

    cases = build_cases(
        "test", sources.speech("test"), sources.noises("test", ("white",)), (0.0, 10.0)
    )
    case_id = cases[0].case_id
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(fake_results(case_id)))
    performance = tmp_path / "perf.json"
    performance.write_text(
        json.dumps(
            {"algorithmic_delay_ms": 48.0, "timing": [{"block": 128, "real_time_factor": 0.1}]}
        )
    )
    files = figures.make_all(results_path, tmp_path / "figs", sources, performance)
    names = {f.name for f in files}
    expected = {
        "fig_delta_vs_snr.png", "fig_ablation.png", "fig_pause_psd.png", "fig_gain_per_band.png",
        "fig_noise_tracking.png", "fig_latency_rtf.png", "fig_example_waveforms.png",
    }  # fmt: skip
    assert expected <= names
    results = json.loads(results_path.read_text())
    assert figures.find_figure_case(results, sources) is not None
    results["protocol"]["figure_case"] = "nonexistent"
    assert figures.find_figure_case(results, sources) is None
    results_path.write_text(json.dumps(results))
    without = figures.make_all(results_path, tmp_path / "figs2", None, performance)
    assert "fig_example_waveforms.png" not in {f.name for f in without}
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.json"
        bad.write_text("[1]")
        figures.load_results(bad)
    assert math.isnan(figures._num(None))


def test_main_reports_missing_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert figures.main(["--results", str(tmp_path / "none.json")]) == 1
    assert "make experiments" in capsys.readouterr().out
