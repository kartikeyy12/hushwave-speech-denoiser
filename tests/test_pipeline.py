from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hushwave import pipeline, tuning
from hushwave.corpus import clear_audio_cache
from hushwave.pipeline import (
    PROFILES,
    Profile,
    TestAlreadyRunError,
    discover,
    stage_real,
    stage_test,
    stage_tune,
    validate_results,
)
from hushwave.systems import make_ablations

FS = 16_000


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = round(seconds * FS)
    t = np.arange(n) / FS
    carrier = sum(np.sin(2 * np.pi * f * t) for f in (140 + 20 * seed, 420, 1100, 2600))
    envelope = np.clip(np.sin(2 * np.pi * 1.3 * t + seed), 0, None) ** 2
    return 0.1 * carrier * envelope + 1e-4 * rng.standard_normal(n)


def _write(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, FS, subtype="PCM_16")


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    clear_audio_cache()
    data = tmp_path / "data"
    entries: dict[str, list[dict[str, object]]] = {"dev": [], "test": []}
    for split, speakers in (("dev", (1, 2, 5)), ("test", (3, 4, 6))):
        for k, sid in enumerate(speakers):
            utt = f"{sid}-0-{k}"
            _write(data / "clean" / split / f"{utt}.wav", _speechlike(3.4, sid))
            entries[split].append(
                {
                    "speaker_id": str(sid),
                    "utterance_id": utt,
                    "path": f"clean/{split}/{utt}.wav",
                    "duration_s": 3.4,
                    "sex": "F" if k % 2 == 0 else "M",
                }
            )
    rng = np.random.default_rng(0)
    _write(data / "noise" / "demand" / "PCAFETER.wav", 0.02 * rng.standard_normal(40 * FS))
    manifest = {
        "schema_version": 1,
        "seed": 1,
        "sources": {},
        **entries,
        "noise": {
            "PCAFETER": {
                "path": "noise/demand/PCAFETER.wav",
                "ranges": {"dev": [0.0, 20.0], "test": [20.0, 40.0]},
            }
        },
    }
    manifest_path = tmp_path / "results" / "split_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest))
    rec = tmp_path / "rec"
    _write(rec / "Fan_noise.wav", 0.03 * rng.standard_normal(40 * FS))
    scene = 0.03 * rng.standard_normal(20 * FS)
    burst = _speechlike(20.0, 9)
    _write(rec / "Speech_in_fan.wav", scene + burst)
    return {"data": data, "manifest": manifest_path, "rec": rec, "results": tmp_path / "results"}


@pytest.fixture
def tiny_profile(monkeypatch: pytest.MonkeyPatch) -> Profile:
    monkeypatch.setattr(tuning, "RULES", ("lsa",))
    monkeypatch.setattr(tuning, "ESTIMATORS", ("mcra",))
    monkeypatch.setattr(tuning, "PREFILTERS", ("none", "spectral"))
    monkeypatch.setattr(tuning, "RISE_RATES", (0.0, 3.0))
    monkeypatch.setattr(tuning, "GAIN_GRID", {"g_min": (0.1, 0.3)})
    monkeypatch.setattr(tuning, "RISE_GRID", (3.0, 6.0))
    monkeypatch.setattr(tuning, "MCRA_DELTA_GRID", (5.0, 3.0))
    monkeypatch.setattr(tuning, "B1_GRID", {"ss_a": (2.0,), "g_min": (0.1,)})
    monkeypatch.setattr(tuning, "B4_GRID", (0.1,))
    monkeypatch.setattr(tuning, "B2_WINDOWS", (15,))
    monkeypatch.setattr("hushwave.pipeline.make_ablations", lambda b5: make_ablations(b5)[:2])
    return Profile("tiny", (5.0,), ("white",), 1, None)


def test_discover_reports_what_is_missing(workspace: dict[str, Path], tmp_path: Path) -> None:
    sources = discover(workspace["data"], workspace["rec"], workspace["manifest"])
    assert sources.has_libri
    assert sorted(sources.recordings) == ["fan_noise", "speech_in_fan"]
    assert any("corridor_noise" in m for m in sources.missing)
    empty = discover(tmp_path / "nodata", tmp_path / "norec", tmp_path / "nomanifest.json")
    assert not empty.has_libri and len(empty.missing) >= 6
    (workspace["data"] / "noise" / "demand" / "PCAFETER.wav").unlink()
    broken = discover(workspace["data"], workspace["rec"], workspace["manifest"])
    assert not broken.has_libri and "missing" in broken.missing[0]


def test_stages_end_to_end_and_the_test_lock(
    workspace: dict[str, Path], tiny_profile: Profile
) -> None:
    sources = discover(workspace["data"], workspace["rec"], workspace["manifest"])
    results_dir = workspace["results"]
    tuned = stage_tune(sources, tiny_profile, 1, results_dir)
    assert (results_dir / "tuning.json").exists()
    assert stage_tune(sources, tiny_profile, 1, results_dir) == tuned  # kept, not redone
    results = stage_test(sources, tiny_profile, 1, results_dir)
    assert results["protocol"]["test_evaluations"] == 1
    assert results["meta"]["complete"] is False  # recordings missing in this workspace
    test = results["test"]
    assert test["primary_corpus"] == "libri" and test["n_utterances"] == 3
    assert {"B0", "B1", "B2", "B3", "B4", "B5"} <= {s["name"] for s in test["systems"]}
    assert sum(s["name"].startswith("A:") for s in test["systems"]) == 2
    libri = test["tables"]["libri"]["overall"]
    assert libri["B0"]["delta_snr_db"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert libri["B5"]["delta_snr_db"]["mean"] is not None
    assert (results_dir / "test_cases.csv.gz").stat().st_size > 0
    with gzip.open(results_dir / "test_cases.csv.gz", "rt") as handle:
        header = handle.readline().strip().split(",")
    assert {"case", "system", "delta_snr_db", "pesq_out"} <= set(header)
    assert validate_results(json.loads((results_dir / "results.json").read_text())) == []
    with pytest.raises(TestAlreadyRunError):
        stage_test(sources, tiny_profile, 1, results_dir)
    again = stage_test(sources, tiny_profile, 1, results_dir, force=True)
    history = again["protocol"]["test_history"]
    assert len(history) == 2 and history[1]["forced"] is True
    real = stage_real(sources, tiny_profile, results_dir)
    assert real is not None
    scene = real["real_recordings"]["scenes"]["speech_in_fan"]
    assert scene["systems"]["B5"]["noise_reduction_pause_db"] > 0.0
    assert (results_dir / "audio" / "speech_in_fan_B5.wav").exists()
    assert (
        json.loads((results_dir / "results.json").read_text())["test"]["n_cases"] == test["n_cases"]
    )


def test_stage_errors_and_skips(workspace: dict[str, Path], tiny_profile: Profile) -> None:
    sources = discover(workspace["data"], workspace["rec"], workspace["manifest"])
    with pytest.raises(RuntimeError, match="tune stage"):
        stage_test(sources, tiny_profile, 1, workspace["results"])
    with pytest.raises(RuntimeError, match="tune stage"):
        stage_real(sources, tiny_profile, workspace["results"])
    stage_tune(sources, tiny_profile, 1, workspace["results"])
    no_real = discover(workspace["data"], workspace["rec"] / "none", workspace["manifest"])
    assert stage_real(no_real, tiny_profile, workspace["results"]) is None
    no_data = discover(workspace["data"] / "x", workspace["rec"] / "none", workspace["manifest"])
    with pytest.raises(RuntimeError, match="no DEV speech"):
        stage_tune(no_data, tiny_profile, 1, workspace["results"] / "other", retune=True)


def test_validate_results_names_the_problem() -> None:
    assert "missing section 'meta'" in validate_results({"schema_version": 2})
    good = {
        "schema_version": 2,
        "meta": {},
        "tuning": {},
        "protocol": {"test_evaluations": 0},
        "test": {
            "systems": [{"name": n} for n in pipeline.SYSTEM_ORDER[:-1]],
            "primary_corpus": "libri",
            "tables": {"libri": {"overall": {}}},
            "tests_vs_baselines": {"libri": [{"p_utterance_level": 1.5}]},
        },
    }
    problems = validate_results(good)
    assert any("test_evaluations" in p for p in problems)
    assert any("lacks B5" in p for p in problems)
    assert any("no finite delta_snr_db mean for B0" in p for p in problems)
    assert any("out of range" in p for p in problems)
    assert any("schema_version" in p for p in validate_results({"schema_version": 1}))


def test_profiles_and_main(
    workspace: dict[str, Path],
    tiny_profile: Profile,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert set(PROFILES) == {"full", "quick"} and PROFILES["full"].tuning_per_cell == 4
    monkeypatch.setitem(PROFILES, "tiny", tiny_profile)
    base = [
        "--profile", "tiny", "--workers", "1",
        "--data-dir", str(workspace["data"]), "--recordings", str(workspace["rec"]),
        "--manifest", str(workspace["manifest"]), "--results-dir", str(workspace["results"]),
    ]  # fmt: skip
    assert pipeline.main(["--stage", "all", *base]) == 0
    assert pipeline.main(["--stage", "test", *base]) == 1
    assert "already evaluated" in capsys.readouterr().err
    assert pipeline.main(["--stage", "test", "--force-retest", *base]) == 0
    provenance = pipeline.provenance()
    assert provenance["packages"]["numpy"] != "not installed" and provenance["cpu_count"]
