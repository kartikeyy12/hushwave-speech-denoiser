"""LibriSpeech scanning, the speaker-disjoint DEV/TEST split, and the split manifest."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SCHEMA_VERSION = 1
MIN_TOTAL_UTTERANCES = 30
MIN_TOTAL_SPEAKERS = 5


@dataclass(frozen=True)
class Speaker:
    speaker_id: str
    sex: str


@dataclass(frozen=True)
class Utterance:
    speaker_id: str
    utterance_id: str
    path: str
    duration_s: float


@dataclass(frozen=True)
class Split:
    dev: tuple[Utterance, ...]
    test: tuple[Utterance, ...]
    sexes: Mapping[str, str]


def parse_speakers(text: str) -> dict[str, Speaker]:
    """Parse LibriSpeech SPEAKERS.TXT lines of the form `ID | SEX | SUBSET | MINUTES | NAME`."""
    speakers: dict[str, Speaker] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 2 or fields[1] not in ("F", "M"):
            raise ValueError(f"cannot parse speaker line: {line!r}")
        speakers[fields[0]] = Speaker(fields[0], fields[1])
    return speakers


def find_speakers_file(root: Path) -> Path:
    """Locate SPEAKERS.TXT under `root` (it ships inside the LibriSpeech archives)."""
    matches = sorted(root.rglob("SPEAKERS.TXT"))
    if not matches:
        raise FileNotFoundError(f"SPEAKERS.TXT not found under {root}")
    return matches[0]


def scan_librispeech(subset_dir: Path) -> list[Utterance]:
    """All FLAC utterances in <subset_dir>/<speaker>/<chapter>/*.flac with their durations."""
    utterances: list[Utterance] = []
    for path in sorted(subset_dir.glob("*/*/*.flac")):
        info = sf.info(str(path))
        utterances.append(
            Utterance(
                speaker_id=path.parts[-3],
                utterance_id=path.stem,
                path=str(path),
                duration_s=float(info.frames) / float(info.samplerate),
            )
        )
    if not utterances:
        raise FileNotFoundError(f"no .flac utterances found under {subset_dir}")
    return utterances


def build_split(
    utterances: Sequence[Utterance],
    speakers: Mapping[str, Speaker],
    dev_speakers: int = 4,
    test_speakers: int = 6,
    utterances_per_speaker: int = 4,
    min_duration_s: float = 3.0,
    max_duration_s: float = 8.0,
    seed: int = 20260101,
) -> Split:
    """Speaker-disjoint, gender-balanced DEV/TEST selection. Deterministic for a given seed."""
    if dev_speakers % 2 or test_speakers % 2:
        raise ValueError("speaker counts must be even so each split is gender-balanced")
    usable: dict[str, list[Utterance]] = {}
    for utt in utterances:
        if utt.speaker_id in speakers and min_duration_s <= utt.duration_s <= max_duration_s:
            usable.setdefault(utt.speaker_id, []).append(utt)
    rng = np.random.default_rng(seed)
    pools: dict[str, list[str]] = {"F": [], "M": []}
    for speaker_id in sorted(usable):
        if len(usable[speaker_id]) >= utterances_per_speaker:
            pools[speakers[speaker_id].sex].append(speaker_id)
    need = dev_speakers // 2 + test_speakers // 2
    for sex, pool in pools.items():
        if len(pool) < need:
            raise ValueError(f"need {need} usable {sex} speakers, found {len(pool)}")
        rng.shuffle(pool)

    def choose(speaker_ids: list[str]) -> tuple[Utterance, ...]:
        chosen: list[Utterance] = []
        for speaker_id in sorted(speaker_ids):
            candidates = sorted(usable[speaker_id], key=lambda u: u.utterance_id)
            picks = rng.choice(len(candidates), size=utterances_per_speaker, replace=False)
            chosen.extend(candidates[int(i)] for i in sorted(picks))
        return tuple(chosen)

    dev_each, test_each = dev_speakers // 2, test_speakers // 2
    dev_ids = pools["F"][:dev_each] + pools["M"][:dev_each]
    test_ids = (
        pools["F"][dev_each : dev_each + test_each] + pools["M"][dev_each : dev_each + test_each]
    )
    sexes = {sid: speakers[sid].sex for sid in dev_ids + test_ids}
    return Split(dev=choose(dev_ids), test=choose(test_ids), sexes=sexes)


def noise_ranges(duration_s: float) -> dict[str, list[float]]:
    """DEV tunes on the first half of each noise recording, TEST evaluates on the second half."""
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    half = duration_s / 2.0
    return {"dev": [0.0, half], "test": [half, duration_s]}


def _entry(utt: Utterance, sexes: Mapping[str, str], root: Path) -> dict[str, Any]:
    entry = asdict(utt)
    entry["path"] = str(Path(utt.path).relative_to(root))
    entry["sex"] = sexes[utt.speaker_id]
    entry["duration_s"] = round(utt.duration_s, 3)
    return entry


def build_manifest(
    split: Split,
    root: Path,
    seed: int,
    noise_files: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON-serialisable split manifest (paths relative to `root`)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "sources": dict(sources),
        "dev": [_entry(u, split.sexes, root) for u in split.dev],
        "test": [_entry(u, split.sexes, root) for u in split.test],
        "noise": {name: dict(info) for name, info in noise_files.items()},
    }


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return a list of problems (empty when the manifest satisfies every rule)."""
    problems: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    splits: dict[str, list[dict[str, Any]]] = {}
    for name in ("dev", "test"):
        entries = manifest.get(name)
        if not isinstance(entries, list) or not entries:
            problems.append(f"{name} must be a non-empty list")
            continue
        splits[name] = entries
        for index, entry in enumerate(entries):
            missing = {"speaker_id", "utterance_id", "path", "duration_s", "sex"} - set(entry)
            if missing:
                problems.append(f"{name}[{index}] is missing {sorted(missing)}")
    if len(splits) == 2:
        speakers = {n: {e.get("speaker_id") for e in entries} for n, entries in splits.items()}
        overlap = speakers["dev"] & speakers["test"]
        if overlap:
            problems.append(f"speakers appear in both DEV and TEST: {sorted(map(str, overlap))}")
        for name, entries in splits.items():
            if {e.get("sex") for e in entries} != {"F", "M"}:
                problems.append(f"{name} must contain both genders")
        if len(speakers["dev"] | speakers["test"]) < MIN_TOTAL_SPEAKERS:
            problems.append(f"need at least {MIN_TOTAL_SPEAKERS} speakers in total")
        total = len(splits["dev"]) + len(splits["test"])
        if total < MIN_TOTAL_UTTERANCES:
            problems.append(f"need at least {MIN_TOTAL_UTTERANCES} utterances, found {total}")
    noise = manifest.get("noise")
    if not isinstance(noise, dict) or not noise:
        problems.append("noise must list at least one recording")
    else:
        for name, info in noise.items():
            ranges = info.get("ranges", {})
            dev_end = ranges.get("dev", [0.0, 0.0])[1]
            test_start = ranges.get("test", [0.0, 0.0])[0]
            if not dev_end <= test_start:
                problems.append(f"noise {name}: DEV and TEST time ranges overlap")
    return problems


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError("invalid manifest: " + "; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("manifest must be a JSON object")
    return loaded
