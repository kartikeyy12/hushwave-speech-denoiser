"""Speaker parsing, the speaker-disjoint split, and manifest validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from hushwave.dataset import (
    Speaker,
    Utterance,
    build_manifest,
    build_split,
    find_speakers_file,
    noise_ranges,
    parse_speakers,
    read_manifest,
    scan_librispeech,
    validate_manifest,
    write_manifest,
)

SPEAKER_TEXT = """;ID  |SEX| SUBSET           |MINUTES| NAME
;
14   | F | train-clean-360  | 25.03 | Kristin LeMoine
16   | F | train-clean-360  | 25.11 | Alys AtHome
19   | M | train-clean-100  | 25.19 | Some Name
"""


def _synthetic(
    n_speakers: int = 14, per_speaker: int = 6
) -> tuple[list[Utterance], dict[str, Speaker]]:
    speakers = {
        str(200 + i): Speaker(str(200 + i), "F" if i % 2 == 0 else "M") for i in range(n_speakers)
    }
    utterances = [
        Utterance(s, f"{s}-1-{k:04d}", f"/x/{s}/{k}.flac", 3.0 + 0.5 * k)
        for s in speakers
        for k in range(per_speaker)
    ]
    utterances.append(Utterance("200", "200-1-9999", "/x/long.flac", 30.0))
    utterances.append(Utterance("999", "999-1-0000", "/x/unknown.flac", 5.0))
    return utterances, speakers


def test_parse_speakers_skips_comments_and_reads_sex() -> None:
    speakers = parse_speakers(SPEAKER_TEXT)
    assert speakers["14"] == Speaker("14", "F")
    assert speakers["19"].sex == "M"
    assert len(speakers) == 3


def test_parse_speakers_rejects_malformed_lines() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        parse_speakers("14 | X | subset | 1.0 | Name")
    with pytest.raises(ValueError):
        parse_speakers("just some words")


def test_find_speakers_file(fake_librispeech: Path, tmp_path: Path) -> None:
    assert find_speakers_file(fake_librispeech).name == "SPEAKERS.TXT"
    with pytest.raises(FileNotFoundError):
        find_speakers_file(tmp_path / "empty_dir_that_does_not_exist")


def test_scan_librispeech_reads_durations(fake_librispeech: Path) -> None:
    found = scan_librispeech(fake_librispeech / "LibriSpeech" / "dev-clean")
    assert len(found) == 60
    assert {u.speaker_id for u in found} == {str(100 + i) for i in range(12)}
    durations = sorted({round(u.duration_s, 1) for u in found})
    assert durations == [1.0, 3.2, 3.4, 3.6, 3.8]
    with pytest.raises(FileNotFoundError):
        scan_librispeech(fake_librispeech / "LibriSpeech" / "nothing-here")


def test_split_is_speaker_disjoint_balanced_and_sized() -> None:
    utterances, speakers = _synthetic()
    split = build_split(utterances, speakers)
    dev_ids = {u.speaker_id for u in split.dev}
    test_ids = {u.speaker_id for u in split.test}
    assert dev_ids.isdisjoint(test_ids)
    assert len(dev_ids) == 4 and len(test_ids) == 6
    assert len(split.dev) == 16 and len(split.test) == 24
    for ids in (dev_ids, test_ids):
        sexes = [speakers[i].sex for i in ids]
        assert sexes.count("F") == sexes.count("M")


def test_split_respects_the_duration_window_and_known_speakers() -> None:
    utterances, speakers = _synthetic()
    split = build_split(utterances, speakers, min_duration_s=3.0, max_duration_s=8.0)
    for utt in split.dev + split.test:
        assert 3.0 <= utt.duration_s <= 8.0
        assert utt.speaker_id != "999"


def test_split_is_deterministic_and_seed_dependent() -> None:
    utterances, speakers = _synthetic()
    first = build_split(utterances, speakers, seed=1)
    assert first == build_split(utterances, speakers, seed=1)
    assert first != build_split(utterances, speakers, seed=2)


def test_split_validates_its_inputs() -> None:
    utterances, speakers = _synthetic()
    with pytest.raises(ValueError, match="even"):
        build_split(utterances, speakers, dev_speakers=3)
    with pytest.raises(ValueError, match="usable"):
        build_split(utterances, speakers, dev_speakers=10, test_speakers=10)
    with pytest.raises(ValueError, match="usable"):
        build_split(utterances, speakers, utterances_per_speaker=50)


def test_noise_ranges_do_not_overlap() -> None:
    ranges = noise_ranges(300.0)
    assert ranges == {"dev": [0.0, 150.0], "test": [150.0, 300.0]}
    with pytest.raises(ValueError):
        noise_ranges(0.0)


def _good_manifest(tmp_path: Path) -> dict[str, Any]:
    utterances, speakers = _synthetic()
    split = build_split(utterances, speakers)
    noise = {"STREET": {"path": "n.wav", "duration_s": 300.0, "ranges": noise_ranges(300.0)}}
    return build_manifest(split, Path("/x"), 7, noise, {"speech": {"name": "t"}})


def test_good_manifest_validates(tmp_path: Path) -> None:
    manifest = _good_manifest(tmp_path)
    assert validate_manifest(manifest) == []
    assert manifest["dev"][0]["path"].startswith("200") or manifest["dev"][0]["path"][0].isdigit()


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = _good_manifest(tmp_path)
    target = tmp_path / "out" / "m.json"
    write_manifest(target, manifest)
    assert read_manifest(target) == json.loads(json.dumps(manifest))


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda m: m.update(schema_version=99), "schema_version"),
        (lambda m: m.update(dev=[]), "dev must be"),
        (lambda m: m["test"].append(dict(m["dev"][0])), "both DEV and TEST"),
        (lambda m: m.update(noise={}), "noise must list"),
        (
            lambda m: m["noise"]["STREET"].update(ranges={"dev": [0, 200], "test": [100, 300]}),
            "overlap",
        ),
        (lambda m: m["dev"][0].pop("sex"), "missing"),
        (lambda m: m.update(dev=m["dev"][:1], test=m["test"][:1]), "at least"),
    ],
)
def test_manifest_problems_are_detected(tmp_path: Path, mutate: Any, expected: str) -> None:
    manifest = copy.deepcopy(_good_manifest(tmp_path))
    mutate(manifest)
    problems = validate_manifest(manifest)
    assert any(expected in p for p in problems), problems


def test_single_gender_split_is_flagged(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_good_manifest(tmp_path))
    for entry in manifest["dev"]:
        entry["sex"] = "F"
    assert any("both genders" in p for p in validate_manifest(manifest))


def test_write_manifest_refuses_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid manifest"):
        write_manifest(tmp_path / "bad.json", {"schema_version": 1})


def test_read_manifest_requires_an_object(tmp_path: Path) -> None:
    (tmp_path / "list.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(tmp_path / "list.json")
