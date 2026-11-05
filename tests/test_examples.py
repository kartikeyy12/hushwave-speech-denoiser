"""Replay clips: excerpt choice, shared gain, spectrogram images and the manifest (M9)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from fakes import fake_results
from hushwave.audio_io import read_audio
from hushwave.examples import (
    CLIP_SECONDS,
    DYNAMIC_RANGE_DB,
    build_clip,
    build_examples,
    excerpt_bounds,
    save_spectrogram,
    spectrogram_db,
)
from hushwave.live import ROOT, TUNING_PATH


def _recording(path: Path, seconds: float, rng: np.random.Generator, amplitude: float) -> None:
    t = np.arange(round(16_000 * seconds)) / 16_000
    voiced = np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)
    x = amplitude * (voiced + 0.1 * rng.standard_normal(t.size))
    sf.write(str(path), x, 16_000, subtype="PCM_16")


@pytest.fixture
def recordings(tmp_path: Path, rng: np.random.Generator) -> Path:
    folder = tmp_path / "recordings"
    folder.mkdir()
    _recording(folder / "Speech_in_fan.wav", 20.0, rng, 0.2)
    _recording(folder / "speech_in_corridor.wav", 18.0, rng, 0.2)
    return folder


def test_excerpt_bounds_short_and_long() -> None:
    long = excerpt_bounds(16_000 * 60)
    assert long.stop - long.start == round(CLIP_SECONDS * 16_000)
    assert long.start == round(0.55 * 16_000 * 60)
    short = excerpt_bounds(16_000 * 5)
    assert (short.start, short.stop) == (0, 16_000 * 5)
    edge = excerpt_bounds(16_000 * 15)
    assert edge.stop == 16_000 * 15


def test_spectrogram_shape_and_scale(rng: np.random.Generator) -> None:
    x = 0.1 * rng.standard_normal(16_000)
    db = spectrogram_db(x)
    assert db.shape[0] == 257 and db.shape[1] > 60
    assert np.all(np.isfinite(db))


def test_spectrogram_image_is_8_bit_grey_with_a_fixed_scale(tmp_path: Path) -> None:
    db = np.full((257, 40), -50.0)
    db[100, :] = 0.0
    save_spectrogram(db, 0.0, tmp_path / "s.png")
    image = Image.open(tmp_path / "s.png")
    assert image.mode == "L" and image.size == (40, 257)
    pixels = np.asarray(image)
    assert pixels[257 - 1 - 100, 0] == 0  # loudest is black, low frequencies are at the bottom
    assert pixels[0, 0] == round(255 * (1 - (DYNAMIC_RANGE_DB - 50) / DYNAMIC_RANGE_DB))
    save_spectrogram(np.full((257, 4), -500.0), 0.0, tmp_path / "quiet.png")
    assert int(np.asarray(Image.open(tmp_path / "quiet.png")).min()) == 255


def test_build_examples_writes_clips_images_and_manifest(recordings: Path, tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps(fake_results()), encoding="utf-8")
    out = tmp_path / "examples"
    manifest = build_examples(recordings, out, TUNING_PATH, results)
    assert [c["id"] for c in manifest["clips"]] == ["speech_in_fan", "speech_in_corridor"]
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest
    clip = manifest["clips"][0]
    lengths = set()
    for key, entry in clip["files"].items():
        audio, fs = read_audio(out / entry["audio"])
        assert fs == 16_000 and np.max(np.abs(audio)) <= 0.9 + 1e-4, key
        lengths.add(audio.size)
        assert Image.open(out / entry["image"]).size[1] == 257
    assert lengths == {round(CLIP_SECONDS * 16_000)}
    assert clip["duration_s"] == pytest.approx(CLIP_SECONDS)
    assert set(clip["measurements"]) == {"original", "b1", "b5"}
    assert clip["measurements"]["b5"]["noise_reduction_pause_db"] == 8.0
    corridor = manifest["clips"][1]
    assert corridor["measurements"] == {}  # the fake results only hold one scene


def test_clips_share_one_gain_so_levels_stay_comparable(recordings: Path, tmp_path: Path) -> None:
    out = tmp_path / "examples"
    build_examples(recordings, out, TUNING_PATH, tmp_path / "missing.json", ["speech_in_fan"])
    original, _ = read_audio(out / "speech_in_fan_original.wav")
    b5, _ = read_audio(out / "speech_in_fan_b5.wav")
    assert float(np.sqrt(np.mean(b5**2))) < float(np.sqrt(np.mean(original**2)))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["clips"][0]["measurements"] == {}


def test_loud_input_is_scaled_without_clipping(tmp_path: Path, rng: np.random.Generator) -> None:
    folder = tmp_path / "rec"
    folder.mkdir()
    _recording(folder / "speech_in_fan.wav", 16.0, rng, 3.0)  # clipped on disk, still loud
    clip = build_clip(
        "speech_in_fan", folder / "speech_in_fan.wav", tmp_path, TUNING_PATH, tmp_path / "r.json"
    )
    audio, _ = read_audio(tmp_path / clip["files"]["original"]["audio"])
    assert np.max(np.abs(audio)) <= 0.9 + 1e-4


def test_no_usable_recordings_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no usable recordings"):
        build_examples(tmp_path / "none", tmp_path / "out")


def test_default_results_path_is_inside_the_repo() -> None:
    assert ROOT.joinpath("results").is_dir()
