from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hushwave.corpus import (
    SNR_GRID,
    Case,
    NoiseItem,
    SpeechItem,
    build_cases,
    case_seed,
    clear_audio_cache,
    find_recordings,
    libri_items,
    load_speech,
    noise_items,
    normalise_stem,
    own_voice_items,
    realize,
    segment_voice,
    select_tuning_cases,
)
from hushwave.synth import measured_snr_db

FS = 16_000


def _bursty(seconds: float, seed: int = 0) -> np.ndarray:
    """Noise bursts of 1.2 s separated by 0.3 s of near-silence."""
    rng = np.random.default_rng(seed)
    x = 1e-4 * rng.standard_normal(round(seconds * FS))
    period = round(1.5 * FS)
    for start in range(0, x.size, period):
        stop = min(start + round(1.2 * FS), x.size)
        x[start:stop] = 0.1 * rng.standard_normal(stop - start)
    return x


def _write(path: Path, x: np.ndarray, fs: int = FS) -> Path:
    sf.write(str(path), x, fs, subtype="PCM_16")
    clear_audio_cache()
    return path


def test_normalise_stem_and_find_recordings(tmp_path: Path) -> None:
    assert normalise_stem("Speech_in_fan_16k.wav") == "speech_in_fan"
    assert normalise_stem("Speech quiet.WAV") == "speech_quiet"
    for name in ("Fan_noise.wav", "corridor_noise_16k.wav", "unrelated.wav", "note.txt"):
        (tmp_path / name).write_bytes(b"x")
    found = find_recordings(tmp_path)
    assert sorted(found) == ["corridor_noise", "fan_noise"]
    assert find_recordings(tmp_path / "missing") == {}


def test_segment_voice_cuts_in_pauses_and_respects_lengths() -> None:
    x = _bursty(40.0)
    pieces = segment_voice(x, FS, 3.0, 8.0)
    assert len(pieces) >= 5
    for (a, b), (c, _) in zip(pieces, pieces[1:], strict=False):
        assert b == c
        assert 3.0 * FS <= b - a <= 8.0 * FS + 256
    for _, stop in pieces[:-1]:
        assert np.sqrt(np.mean(x[stop - 128 : stop + 128] ** 2)) < 0.02
    with pytest.raises(ValueError):
        segment_voice(x, FS, 5.0, 3.0)
    assert segment_voice(x[: 2 * FS], FS, 3.0, 8.0) == []


def test_own_voice_items_split_without_overlap(tmp_path: Path) -> None:
    path = _write(tmp_path / "speech_quiet.wav", _bursty(60.0, 1))
    dev = own_voice_items(path, "dev")
    test = own_voice_items(path, "test")
    assert dev and test
    assert max(i.stop for i in dev) <= min(i.start for i in test)
    assert all(i.corpus == "own" and i.speaker == "own" for i in dev + test)
    assert len({i.name for i in dev + test}) == len(dev) + len(test)
    assert own_voice_items(path, "dev", min_active=1.01) == []
    with pytest.raises(ValueError):
        own_voice_items(path, "all")
    assert load_speech(dev[0]).size == dev[0].stop - dev[0].start


def test_libri_and_noise_items(tmp_path: Path) -> None:
    manifest = {
        "dev": [{"utterance_id": "1-2-3", "speaker_id": "1", "path": "clean/dev/1-2-3.wav"}],
        "test": [{"utterance_id": "4-5-6", "speaker_id": "4", "path": "clean/test/4-5-6.wav"}],
        "noise": {
            "PCAFETER": {
                "path": "noise/demand/PCAFETER.wav",
                "ranges": {"dev": [0.0, 10.0], "test": [10.0, 20.0]},
            }
        },
    }
    (item,) = libri_items(manifest, tmp_path, "test")
    assert item.name == "4-5-6" and item.path == str(tmp_path / "clean/test/4-5-6.wav")
    with pytest.raises(ValueError):
        libri_items(manifest, tmp_path, "x")
    fan = _write(tmp_path / "Fan_noise.wav", np.random.default_rng(0).standard_normal(8 * FS))
    dev = noise_items(manifest, tmp_path, {"fan_noise": fan}, "dev", synthetic=("white", "babble"))
    test = noise_items(manifest, tmp_path, {"fan_noise": fan}, "test", synthetic=("white",))
    assert [n.name for n in dev] == ["syn-white", "syn-babble", "demand-PCAFETER", "own-fan_noise"]
    demand_dev = next(n for n in dev if n.kind == "demand")
    demand_test = next(n for n in test if n.kind == "demand")
    assert (demand_dev.start, demand_dev.stop) == (0, 10 * FS)
    assert (demand_test.start, demand_test.stop) == (10 * FS, 20 * FS)
    fan_dev = next(n for n in dev if n.kind == "recording")
    fan_test = next(n for n in test if n.kind == "recording")
    assert fan_dev.stop == fan_test.start == 4 * FS
    assert [n.name for n in noise_items(None, tmp_path, {}, "dev", ("pink",))] == ["syn-pink"]
    with pytest.raises(ValueError):
        noise_items(None, tmp_path, {}, "all")


def _speech(n: int) -> list[SpeechItem]:
    return [SpeechItem("libri", f"u{i}", f"s{i // 2}", f"/x/{i}.wav") for i in range(n)]


def test_build_cases_is_complete_deterministic_and_seeded() -> None:
    speech = _speech(6)
    noises = [NoiseItem("synthetic", "syn-white"), NoiseItem("synthetic", "syn-babble")]
    cases = build_cases("test", speech, noises)
    assert len(cases) == 6 * 2 * len(SNR_GRID)
    assert cases == build_cases("test", speech, noises)
    assert len({c.case_id for c in cases}) == len(cases)
    assert len({c.seed for c in cases}) == len(cases)
    assert case_seed("test", "u0", "syn-white", 5.0) == case_seed("test", "u0", "syn-white", 5.0)
    assert case_seed("test", "u0", "syn-white", 5.0) != case_seed("dev", "u0", "syn-white", 5.0)
    for case in cases:
        if case.noise.name == "syn-babble":
            assert len(case.noise.sources) == 4
            assert all(s.speaker != case.speech.speaker for s in case.noise.sources)


def test_babble_needs_other_utterances() -> None:
    with pytest.raises(ValueError, match="babble"):
        build_cases("dev", _speech(1), [NoiseItem("synthetic", "syn-babble")])
    own = [SpeechItem("own", f"o{i}", "own", "/x.wav") for i in range(3)]
    (case, *_) = build_cases("dev", own, [NoiseItem("synthetic", "syn-babble")], (0.0,))
    assert all(s.name != case.speech.name for s in case.noise.sources)


def test_select_tuning_cases_covers_every_cell() -> None:
    noises = [NoiseItem("synthetic", f"syn-{k}") for k in ("white", "pink")]
    cases = build_cases("dev", _speech(8), noises)
    chosen = select_tuning_cases(cases, 3)
    assert len(chosen) == 2 * len(SNR_GRID) * 3
    assert {(c.noise.name, c.snr_db) for c in chosen} == {(c.noise.name, c.snr_db) for c in cases}
    assert len(select_tuning_cases(cases, 100)) == len(cases)
    with pytest.raises(ValueError):
        select_tuning_cases(cases, 0)


def test_realize_hits_the_requested_snr_for_every_noise_kind(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.wav", _bursty(5.0, 1))
    b = _write(tmp_path / "b.wav", _bursty(5.0, 2))
    noise_file = _write(tmp_path / "n.wav", np.random.default_rng(3).standard_normal(20 * FS))
    target = SpeechItem("libri", "t", "s1", str(a))
    other = [
        SpeechItem("libri", "o1", "s2", str(b)),
        SpeechItem("libri", "o2", "s3", str(b), 0, 2 * FS),
    ]
    noises = [
        NoiseItem("synthetic", "syn-hum"),
        NoiseItem("synthetic", "syn-babble", sources=tuple(other)),
        NoiseItem("demand", "demand-X", str(noise_file), 0, 10 * FS),
        NoiseItem("recording", "own-fan_noise", str(noise_file), 10 * FS, 0),
    ]
    for noise in noises:
        case = Case("test", target, noise, 5.0, case_seed("test", "t", noise.name, 5.0))
        first = realize(case)
        again = realize(case)
        np.testing.assert_array_equal(first.noisy, again.noisy)
        assert measured_snr_db(first.clean, first.noise) == pytest.approx(5.0, abs=1e-9)
        np.testing.assert_allclose(first.noisy, first.clean + first.noise)
