"""One-command data preparation: download, verify, extract, split, and write the manifest.

Sources (checked while building milestone M3):
- LibriSpeech dev-clean, OpenSLR SLR12, license CC BY 4.0. The archive MD5 is read from the
  publisher's md5sum.txt at download time and the download is rejected on mismatch.
- DEMAND noise database, Zenodo record 10.5281/zenodo.1227121. The MD5 values below are copied
  from that record's file list; downloads are rejected on mismatch. Licensing is stated
  inconsistently on the record (see docs/DATA_SOURCES.md); we attribute and never redistribute.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from hushwave.audio_io import TARGET_FS, read_audio, write_wav
from hushwave.dataset import (
    Split,
    Utterance,
    build_manifest,
    build_split,
    find_speakers_file,
    noise_ranges,
    parse_speakers,
    scan_librispeech,
    write_manifest,
)
from hushwave.downloads import (
    check_url,
    download,
    extract_tar,
    extract_zip_member,
    fetch_text,
    md5_file,
    parse_md5sum,
    sha256_file,
)

DEFAULT_SEED = 20260101
Progress = Callable[[str, int, int | None], None]


@dataclass(frozen=True)
class DemandFile:
    name: str
    url: str
    md5: str


@dataclass(frozen=True)
class Sources:
    libri_url: str
    libri_md5sum_url: str
    libri_subset: str
    demand: tuple[DemandFile, ...]


def _demand(name: str, md5: str) -> DemandFile:
    url = f"https://zenodo.org/records/1227121/files/{name}_16k.zip?download=1"
    return DemandFile(name, url, md5)


DEFAULT_SOURCES = Sources(
    libri_url="https://www.openslr.org/resources/12/dev-clean.tar.gz",
    libri_md5sum_url="https://www.openslr.org/resources/12/md5sum.txt",
    libri_subset="dev-clean",
    demand=(
        _demand("STRAFFIC", "2efa87262f272bbf9ba578088e81939c"),
        _demand("PCAFETER", "99927d148128254141a9417d051510bb"),
        _demand("OHALLWAY", "fe918bbb0e63e73d09ba7f4843ef33f1"),
    ),
)


def prepare_librispeech(
    sources: Sources,
    data_dir: Path,
    libri_md5: str | None = None,
    progress: Progress | None = None,
) -> tuple[Path, str]:
    """Download and extract LibriSpeech; returns (extraction root, verified archive MD5)."""
    archive_name = sources.libri_url.rsplit("/", 1)[-1]
    expected = libri_md5
    if expected is None:
        expected = parse_md5sum(fetch_text(sources.libri_md5sum_url)).get(archive_name)
    if expected is None:
        raise RuntimeError(
            f"no MD5 for {archive_name} in {sources.libri_md5sum_url}; "
            "pass the published value with --libri-md5"
        )
    raw = data_dir / "raw"
    archive = raw / archive_name

    def report(done: int, total: int | None) -> None:
        if progress is not None:
            progress(archive_name, done, total)

    download(sources.libri_url, archive, expected, report)
    if not (raw / "LibriSpeech" / sources.libri_subset).is_dir():
        extract_tar(archive, raw)
    return raw, expected


def build_clean_set(raw_root: Path, subset: str, data_dir: Path, seed: int = DEFAULT_SEED) -> Split:
    """Select the DEV/TEST utterances and copy them to data/clean/{dev,test}/ as 16-bit WAV."""
    speakers = parse_speakers(find_speakers_file(raw_root).read_text(encoding="utf-8"))
    utterances = scan_librispeech(raw_root / "LibriSpeech" / subset)
    chosen = build_split(utterances, speakers, seed=seed)

    def copy_all(name: str, items: Sequence[Utterance]) -> tuple[Utterance, ...]:
        out: list[Utterance] = []
        for utt in items:
            samples, _ = read_audio(utt.path, TARGET_FS)
            target = data_dir / "clean" / name / f"{utt.utterance_id}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            write_wav(target, samples, TARGET_FS)
            out.append(Utterance(utt.speaker_id, utt.utterance_id, str(target), utt.duration_s))
        return tuple(out)

    return Split(
        dev=copy_all("dev", chosen.dev), test=copy_all("test", chosen.test), sexes=chosen.sexes
    )


def prepare_demand(
    sources: Sources, data_dir: Path, progress: Progress | None = None
) -> dict[str, dict[str, Any]]:
    """Download each DEMAND category, keep channel 1 only, and describe it for the manifest."""
    info: dict[str, dict[str, Any]] = {}
    for item in sources.demand:
        archive = data_dir / "raw" / "demand" / f"{item.name}_16k.zip"

        def report(done: int, total: int | None, label: str = item.name) -> None:
            if progress is not None:
                progress(label, done, total)

        download(item.url, archive, item.md5, report)
        target = data_dir / "noise" / "demand" / f"{item.name}.wav"
        member = extract_zip_member(archive, "ch01.wav", target)
        header = sf.info(str(target))
        if header.samplerate != TARGET_FS:
            raise RuntimeError(f"{target}: expected {TARGET_FS} Hz, got {header.samplerate}")
        duration = float(header.frames) / float(header.samplerate)
        info[item.name] = {
            "path": str(target.relative_to(data_dir)),
            "archive_member": member,
            "archive_md5": item.md5,
            "duration_s": round(duration, 3),
            "ranges": noise_ranges(duration),
        }
    return info


def write_checksums(path: Path, archives: dict[str, tuple[str, Path]]) -> None:
    """Record url, md5, sha256 and size of every downloaded archive."""
    record = {
        name: {
            "url": url,
            "md5": md5_file(file),
            "sha256": sha256_file(file),
            "bytes": file.stat().st_size,
        }
        for name, (url, file) in sorted(archives.items())
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    sources: Sources,
    data_dir: Path,
    results_dir: Path,
    libri_md5: str | None = None,
    seed: int = DEFAULT_SEED,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Prepare everything and write results/split_manifest.json and data_checksums.json."""
    raw_root, verified_md5 = prepare_librispeech(sources, data_dir, libri_md5, progress)
    split = build_clean_set(raw_root, sources.libri_subset, data_dir, seed)
    noise = prepare_demand(sources, data_dir, progress)
    manifest = build_manifest(
        split,
        data_dir,
        seed,
        noise,
        {
            "speech": {
                "name": "LibriSpeech dev-clean (OpenSLR SLR12)",
                "url": sources.libri_url,
                "license": "CC BY 4.0",
                "archive_md5": verified_md5,
            },
            "noise": {
                "name": "DEMAND (Zenodo 10.5281/zenodo.1227121), channel 1, 16 kHz",
                "license": "see docs/DATA_SOURCES.md",
            },
        },
    )
    write_manifest(results_dir / "split_manifest.json", manifest)
    archives = {
        sources.libri_url.rsplit("/", 1)[-1]: (
            sources.libri_url,
            data_dir / "raw" / sources.libri_url.rsplit("/", 1)[-1],
        )
    }
    for item in sources.demand:
        archives[f"{item.name}_16k.zip"] = (
            item.url,
            data_dir / "raw" / "demand" / f"{item.name}_16k.zip",
        )
    write_checksums(results_dir / "data_checksums.json", archives)
    return manifest


def _print_progress(label: str, done: int, total: int | None) -> None:
    if total:
        sys.stdout.write(f"\r  {label}: {100.0 * done / total:5.1f} %")
    else:
        sys.stdout.write(f"\r  {label}: {done / 1e6:.1f} MB")
    sys.stdout.flush()


def main(argv: Sequence[str], sources: Sources = DEFAULT_SOURCES) -> int:
    parser = argparse.ArgumentParser(description="Download and prepare the HushWave datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--libri-md5", default=None, help="override the published LibriSpeech MD5")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--check-urls", action="store_true", help="only test that URLs resolve")
    args = parser.parse_args(list(argv))

    if args.check_urls:
        urls = [sources.libri_url, sources.libri_md5sum_url] + [d.url for d in sources.demand]
        failures = 0
        for url in urls:
            result = check_url(url)
            size = f"{result.size / 1e6:.1f} MB" if result.size else "size unknown"
            print(f"{result.status}  {size:>14}  {url}")
            failures += 0 if 200 <= result.status < 400 else 1
        print("all URLs resolve" if failures == 0 else f"{failures} URL(s) FAILED")
        return 0 if failures == 0 else 1

    manifest = run(
        sources, args.data_dir, args.results_dir, args.libri_md5, args.seed, _print_progress
    )
    print()
    print(
        f"DEV: {len(manifest['dev'])} utterances, TEST: {len(manifest['test'])} utterances, "
        f"noise recordings: {', '.join(sorted(manifest['noise']))}"
    )
    print(f"wrote {args.results_dir / 'split_manifest.json'} and data_checksums.json")
    return 0
