"""Speech and noise sets for the experiments, and the deterministic list of test cases.

Speech corpora: ``libri`` (LibriSpeech, speaker-disjoint DEV/TEST from the split manifest) and
``own`` (the group's own voice recording: its first half is cut into DEV segments and its second
half into TEST segments, so no sample is shared; one speaker, so it is reported separately).

Noise: synthetic generators, DEMAND categories and the group's own noise recordings. Recordings
and DEMAND are split in time (first half DEV, second half TEST).

Every case carries its own seed, derived from its identity, so results do not depend on the order
in which cases run or on how many worker processes there are.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hushwave.audio_io import TARGET_FS, read_audio
from hushwave.metrics import speech_activity
from hushwave.synth import NOISE_KINDS, make_noise, mix_at_snr, noise_segment

FloatArray = NDArray[np.float64]
RECORDING_ROLES = (
    "fan_noise",
    "corridor_noise",
    "speech_quiet",
    "speech_in_fan",
    "speech_in_corridor",
)
SNR_GRID = (-5.0, 0.0, 5.0, 10.0, 15.0)
BABBLE_SOURCES = 4
OWN_SEGMENT_MIN_S = 3.0
OWN_SEGMENT_MAX_S = 8.0
OWN_MIN_ACTIVE = 0.3
SEGMENT_BLOCK = 256


@dataclass(frozen=True)
class SpeechItem:
    corpus: str
    name: str
    speaker: str
    path: str
    start: int = 0
    stop: int = 0  # 0 means "to the end of the file"


@dataclass(frozen=True)
class NoiseItem:
    kind: str  # "synthetic", "demand" or "recording"
    name: str
    path: str = ""
    start: int = 0
    stop: int = 0
    sources: tuple[SpeechItem, ...] = ()  # babble only


@dataclass(frozen=True)
class Case:
    split: str
    speech: SpeechItem
    noise: NoiseItem
    snr_db: float
    seed: int

    @property
    def case_id(self) -> str:
        return f"{self.split}|{self.speech.name}|{self.noise.name}|{self.snr_db:g}"


@dataclass(frozen=True)
class Realized:
    clean: FloatArray
    noise: FloatArray
    noisy: FloatArray


# --------------------------------------------------------------------------- audio access
@lru_cache(maxsize=24)
def _load_file(path: str) -> FloatArray:
    samples, _ = read_audio(path, TARGET_FS, allow_resample=True)
    samples.setflags(write=False)
    return samples


def clear_audio_cache() -> None:
    _load_file.cache_clear()


def load_speech(item: SpeechItem) -> FloatArray:
    data = _load_file(item.path)
    stop = item.stop if item.stop > 0 else data.size
    return np.array(data[item.start : stop], dtype=np.float64)


def _noise_slice(item: NoiseItem) -> FloatArray:
    data = _load_file(item.path)
    stop = item.stop if item.stop > 0 else data.size
    return np.asarray(data[item.start : stop], dtype=np.float64)


# --------------------------------------------------------------------------- recordings
def normalise_stem(name: str) -> str:
    """'Speech_in_fan_16k.wav' -> 'speech_in_fan'."""
    stem = Path(name).stem.lower().strip()
    stem = re.sub(r"[\s\-]+", "_", stem)
    return re.sub(r"_16k$", "", stem)


def find_recordings(directory: Path) -> dict[str, Path]:
    """Map each recording role (see RECORDING_ROLES) to a file in ``directory``, if present.

    Names are matched case-insensitively and a trailing ``_16k`` is ignored, so both the raw
    recordings and the output of prepare_recordings.py work.
    """
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".wav", ".flac"):
            continue
        role = normalise_stem(path.name)
        if role in RECORDING_ROLES and role not in found:
            found[role] = path
    return found


def segment_voice(
    x: FloatArray,
    fs: int = TARGET_FS,
    min_s: float = OWN_SEGMENT_MIN_S,
    max_s: float = OWN_SEGMENT_MAX_S,
) -> list[tuple[int, int]]:
    """Cut a long speech recording into pieces of min_s to max_s seconds at its quietest points.

    Energy is measured per block of SEGMENT_BLOCK samples and smoothed over five blocks (about
    80 ms); each cut falls on the quietest block between min_s and max_s after the previous cut.
    A trailing piece shorter than min_s is dropped. Returns (start, stop) sample indices.
    """
    if min_s <= 0.0 or max_s < min_s:
        raise ValueError("need 0 < min_s <= max_s")
    count = x.size // SEGMENT_BLOCK
    energy = np.sum(x[: count * SEGMENT_BLOCK].reshape(count, SEGMENT_BLOCK) ** 2, axis=1)
    smooth = np.convolve(energy, np.ones(5) / 5.0, mode="same")
    low = math.ceil(min_s * fs / SEGMENT_BLOCK)
    high = math.floor(max_s * fs / SEGMENT_BLOCK)
    pieces: list[tuple[int, int]] = []
    start = 0
    while count - start >= low:
        if count - start <= high:
            end = count
        else:
            window = smooth[start + low : start + high + 1]
            end = start + low + int(np.argmin(window))
        pieces.append((start * SEGMENT_BLOCK, end * SEGMENT_BLOCK))
        start = end
    return pieces


def own_voice_items(path: Path, split: str, min_active: float = OWN_MIN_ACTIVE) -> list[SpeechItem]:
    """DEV (first half of the file) or TEST (second half) segments of the own-voice recording."""
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")
    data = _load_file(str(path))
    half = data.size // 2
    offset = 0 if split == "dev" else half
    part = data[:half] if split == "dev" else data[half:]
    items: list[SpeechItem] = []
    for index, (start, stop) in enumerate(segment_voice(np.asarray(part))):
        piece = np.asarray(part[start:stop])
        if not np.any(piece):
            continue
        if float(np.mean(speech_activity(piece))) < min_active:
            continue
        items.append(
            SpeechItem(
                "own", f"own-{split}-{index:02d}", "own", str(path), offset + start, offset + stop
            )
        )
    return items


def libri_items(manifest: dict[str, Any], data_dir: Path, split: str) -> list[SpeechItem]:
    """LibriSpeech utterances of one split, from results/split_manifest.json."""
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")
    return [
        SpeechItem(
            "libri",
            str(entry["utterance_id"]),
            str(entry["speaker_id"]),
            str(data_dir / entry["path"]),
        )
        for entry in manifest[split]
    ]


def noise_items(
    manifest: dict[str, Any] | None,
    data_dir: Path,
    recordings: dict[str, Path],
    split: str,
    synthetic: Sequence[str] = NOISE_KINDS,
) -> list[NoiseItem]:
    """All noise conditions for one split. Babble sources are attached per case, not here."""
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")
    items = [NoiseItem("synthetic", f"syn-{kind}") for kind in synthetic]
    if manifest is not None:
        for name, info in sorted(manifest.get("noise", {}).items()):
            fs = TARGET_FS
            start_s, stop_s = info["ranges"][split]
            items.append(
                NoiseItem(
                    "demand",
                    f"demand-{name}",
                    str(data_dir / info["path"]),
                    int(start_s * fs),
                    int(stop_s * fs),
                )
            )
    for role in ("fan_noise", "corridor_noise"):
        if role in recordings:
            data = _load_file(str(recordings[role]))
            half = data.size // 2
            start, stop = (0, half) if split == "dev" else (half, data.size)
            items.append(NoiseItem("recording", f"own-{role}", str(recordings[role]), start, stop))
    return items


# --------------------------------------------------------------------------- cases
def case_seed(split: str, speech: str, noise: str, snr_db: float) -> int:
    """A stable 32-bit seed from the identity of a case (independent of run order)."""
    return zlib.crc32(f"{split}|{speech}|{noise}|{snr_db:g}".encode()) & 0xFFFFFFFF


def _babble_sources(
    target: SpeechItem, pool: Sequence[SpeechItem], shift: int
) -> tuple[SpeechItem, ...]:
    others = [s for s in pool if s.speaker != target.speaker]
    if len(others) < 2:
        others = [s for s in pool if s.name != target.name]
    if len(others) < 2:
        raise ValueError("babble needs at least two other utterances in the same split")
    count = min(BABBLE_SOURCES, len(others))
    return tuple(others[(shift + 3 * j) % len(others)] for j in range(count))


def build_cases(
    split: str,
    speech: Sequence[SpeechItem],
    noises: Sequence[NoiseItem],
    snrs: Sequence[float] = SNR_GRID,
) -> list[Case]:
    """Every (utterance, noise, SNR) combination, in a fixed order, with its seed."""
    cases: list[Case] = []
    for index, utt in enumerate(speech):
        for noise in noises:
            item = noise
            if noise.kind == "synthetic" and noise.name == "syn-babble":
                item = NoiseItem(
                    noise.kind, noise.name, sources=_babble_sources(utt, speech, index)
                )
            for snr in snrs:
                cases.append(
                    Case(split, utt, item, float(snr), case_seed(split, utt.name, noise.name, snr))
                )
    return cases


def select_tuning_cases(cases: Sequence[Case], per_cell: int) -> list[Case]:
    """For every (noise, SNR) cell keep ``per_cell`` cases, spreading over the utterances."""
    if per_cell < 1:
        raise ValueError("per_cell must be at least 1")
    cells: dict[tuple[str, float], list[Case]] = {}
    for case in cases:
        cells.setdefault((case.noise.name, case.snr_db), []).append(case)
    chosen: list[Case] = []
    for number, key in enumerate(sorted(cells)):
        members = cells[key]
        for j in range(min(per_cell, len(members))):
            chosen.append(members[(number + j * max(1, len(members) // per_cell)) % len(members)])
    return chosen


def realize(case: Case) -> Realized:
    """Load the speech, generate or cut the noise, and mix at the case's SNR."""
    clean = load_speech(case.speech)
    rng = np.random.default_rng(case.seed)
    noise = case.noise
    if noise.kind == "synthetic":
        kind = noise.name.removeprefix("syn-")
        sources = [load_speech(s) for s in noise.sources] if noise.sources else None
        raw = make_noise(kind, clean.size, TARGET_FS, rng, sources)
    else:
        raw = noise_segment(_noise_slice(noise), clean.size, rng)
    mixture = mix_at_snr(clean, raw, case.snr_db)
    return Realized(clean=clean, noise=mixture.noise, noisy=mixture.noisy)
