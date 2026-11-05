"""Build the replay clips for the demo (``make examples``).

Each clip is an excerpt of a continuous run: the whole recording goes through the system first
and the excerpt is cut from the outputs afterwards, so the clip shows a system that has been
running for a while, not its start-up transient. Original, B1 and B5 share one gain so that
level differences between them stay audible and nothing clips. The measurements shown next to
a clip are those of the whole recording from ``results/results.json``; none is typed by hand.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from hushwave.audio_io import TARGET_FS, read_audio, write_wav
from hushwave.corpus import find_recordings
from hushwave.live import ROOT, TUNING_PATH, load_b1, load_b5, render
from hushwave.stft import stft

FloatArray = NDArray[np.float64]
SCENES = {
    "speech_in_fan": ("Speech in fan noise", "A voice recorded in front of a running fan."),
    "speech_in_corridor": (
        "Speech in corridor noise",
        "A voice recorded in a corridor with ambient noise.",
    ),
}
VARIANTS = (
    ("original", "B0", "Original"),
    ("b1", "B1", "Spectral subtraction"),
    ("b5", "B5", "HushWave"),
)
CLIP_SECONDS = 14.0
START_FRACTION = 0.55
DYNAMIC_RANGE_DB = 70.0
PEAK = 0.9
FRAME = 512
HOP = 256


def spectrogram_db(x: ArrayLike) -> FloatArray:
    """Power spectrogram in dB, shape (bins, frames), from the project's own STFT."""
    spec = stft(np.asarray(x, dtype=np.float64), FRAME, HOP)
    power = np.abs(spec) ** 2
    return np.asarray(10.0 * np.log10(power + 1e-12).T, dtype=np.float64)


def save_spectrogram(db: FloatArray, top_db: float, path: Path) -> None:
    """8-bit grey image (dark is loud), low frequencies at the bottom, one fixed scale."""
    level = np.clip((db - (top_db - DYNAMIC_RANGE_DB)) / DYNAMIC_RANGE_DB, 0.0, 1.0)
    pixels = np.round(255.0 * (1.0 - level[::-1])).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path, optimize=True)


def excerpt_bounds(n_samples: int, fs: int = TARGET_FS, seconds: float = CLIP_SECONDS) -> slice:
    length = min(n_samples, round(seconds * fs))
    start = min(round(START_FRACTION * n_samples), n_samples - length)
    return slice(start, start + length)


def _measurements(results_path: Path, scene: str) -> dict[str, Any]:
    if not results_path.is_file():
        return {}
    results = json.loads(results_path.read_text(encoding="utf-8"))
    systems = results.get("real_recordings", {}).get("scenes", {}).get(scene, {}).get("systems", {})
    return {
        key: {
            "noise_reduction_pause_db": systems[system]["noise_reduction_pause_db"],
            "speech_level_change_db": systems[system]["speech_level_change_db"],
        }
        for key, system, _ in VARIANTS
        if system in systems
    }


def build_clip(
    scene: str, source: Path, out_dir: Path, tuning_path: Path, results_path: Path
) -> dict[str, Any]:
    x, fs = read_audio(source, TARGET_FS, allow_resample=True)
    outputs = {
        "original": x,
        "b1": render(load_b1(tuning_path), x),
        "b5": render(load_b5(tuning_path), x),
    }
    window = excerpt_bounds(x.size, fs)
    cuts = {key: value[window] for key, value in outputs.items()}
    peak = max(float(np.max(np.abs(v))) for v in cuts.values())
    gain = 1.0 if peak <= PEAK else PEAK / peak
    cuts = {key: value * gain for key, value in cuts.items()}
    specs = {key: spectrogram_db(value) for key, value in cuts.items()}
    top_db = float(np.max(specs["original"]))
    files: dict[str, dict[str, str]] = {}
    for key, _, _ in VARIANTS:
        wav, png = f"{scene}_{key}.wav", f"{scene}_{key}.png"
        write_wav(out_dir / wav, cuts[key], fs)
        save_spectrogram(specs[key], top_db, out_dir / png)
        files[key] = {"audio": wav, "image": png}
    title, note = SCENES[scene]
    return {
        "id": scene,
        "title": title,
        "note": note,
        "duration_s": cuts["original"].size / fs,
        "fs": fs,
        "files": files,
        "measurements": _measurements(results_path, scene),
    }


def build_examples(
    recordings_dir: Path,
    out_dir: Path,
    tuning_path: Path = TUNING_PATH,
    results_path: Path | None = None,
    scenes: Sequence[str] = tuple(SCENES),
) -> dict[str, Any]:
    found = find_recordings(recordings_dir)
    selected = [s for s in scenes if s in found]
    if not selected:
        raise FileNotFoundError(
            f"no usable recordings in {recordings_dir}; expected "
            f"{', '.join(scenes)} as WAV or FLAC files"
        )
    results = results_path if results_path is not None else ROOT / "results" / "results.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = [build_clip(s, found[s], out_dir, tuning_path, results) for s in selected]
    manifest = {
        "variants": [{"key": k, "system": s, "label": label} for k, s, label in VARIANTS],
        "clips": clips,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
