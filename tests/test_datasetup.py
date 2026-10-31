"""End-to-end data preparation against a local server that mimics OpenSLR and Zenodo."""

from __future__ import annotations

import hashlib
import json
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hushwave.dataset import read_manifest, validate_manifest
from hushwave.datasetup import DemandFile, Sources, main, prepare_librispeech, run


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def mirror(
    tmp_path: Path, fake_librispeech: Path, serve: Callable[[Path], str]
) -> tuple[Sources, Path]:
    """Serve a fake dev-clean archive, its md5sum.txt, and three fake DEMAND zips."""
    server = tmp_path / "server"
    server.mkdir()
    archive = server / "dev-clean.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(fake_librispeech / "LibriSpeech", arcname="LibriSpeech")
    (server / "md5sum.txt").write_text(f"{_md5(archive)}  dev-clean.tar.gz\n", encoding="utf-8")
    rng = np.random.default_rng(11)
    demand: list[DemandFile] = []
    base = serve(server)
    for name in ("STREET", "CAFE", "HALL"):
        wav = tmp_path / f"{name}.wav"
        sf.write(str(wav), 0.05 * rng.standard_normal(16_000 * 10), 16_000, subtype="PCM_16")
        bundle = server / f"{name}_16k.zip"
        with zipfile.ZipFile(bundle, "w") as zipped:
            zipped.write(wav, f"{name}/ch01.wav")
            zipped.write(wav, f"{name}/ch02.wav")
        demand.append(DemandFile(name, f"{base}/{bundle.name}", _md5(bundle)))
    sources = Sources(
        libri_url=f"{base}/dev-clean.tar.gz",
        libri_md5sum_url=f"{base}/md5sum.txt",
        libri_subset="dev-clean",
        demand=tuple(demand),
    )
    return sources, server


def test_full_run_builds_a_valid_manifest(tmp_path: Path, mirror: tuple[Sources, Path]) -> None:
    sources, _ = mirror
    data_dir, results_dir = tmp_path / "data", tmp_path / "results"
    manifest = run(sources, data_dir, results_dir)
    assert validate_manifest(manifest) == []
    assert len(manifest["dev"]) == 16 and len(manifest["test"]) == 24
    assert sorted(manifest["noise"]) == ["CAFE", "HALL", "STREET"]
    assert manifest["noise"]["STREET"]["duration_s"] == pytest.approx(10.0)
    assert manifest["noise"]["STREET"]["ranges"]["test"] == [5.0, 10.0]
    for entry in manifest["dev"] + manifest["test"]:
        audio, fs = sf.read(str(data_dir / entry["path"]))
        assert fs == 16_000 and audio.ndim == 1
        assert entry["duration_s"] == pytest.approx(audio.size / fs, abs=1e-3)
    on_disk = read_manifest(results_dir / "split_manifest.json")
    assert on_disk == json.loads(json.dumps(manifest))
    checksums = json.loads((results_dir / "data_checksums.json").read_text(encoding="utf-8"))
    assert set(checksums) == {"dev-clean.tar.gz", "STREET_16k.zip", "CAFE_16k.zip", "HALL_16k.zip"}
    assert all(len(v["sha256"]) == 64 for v in checksums.values())


def test_second_run_reuses_downloads_and_gives_the_same_manifest(
    tmp_path: Path, mirror: tuple[Sources, Path]
) -> None:
    sources, _ = mirror
    first = run(sources, tmp_path / "data", tmp_path / "results")
    labels: list[str] = []
    second = run(
        sources,
        tmp_path / "data",
        tmp_path / "results",
        progress=lambda label, done, total: labels.append(label),
    )
    assert labels == []
    assert first == second


def test_bad_librispeech_checksum_is_refused(tmp_path: Path, mirror: tuple[Sources, Path]) -> None:
    sources, _ = mirror
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        prepare_librispeech(sources, tmp_path / "data", libri_md5="0" * 32)


def test_missing_md5_entry_asks_for_the_override(
    tmp_path: Path, mirror: tuple[Sources, Path]
) -> None:
    sources, server = mirror
    (server / "md5sum.txt").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="--libri-md5"):
        prepare_librispeech(sources, tmp_path / "data")


def test_bad_demand_checksum_is_refused(tmp_path: Path, mirror: tuple[Sources, Path]) -> None:
    sources, _ = mirror
    broken = Sources(
        sources.libri_url,
        sources.libri_md5sum_url,
        sources.libri_subset,
        (DemandFile("STREET", sources.demand[0].url, "f" * 32),),
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        run(broken, tmp_path / "data", tmp_path / "results")


def test_cli_check_urls_reports_success_and_failure(
    tmp_path: Path, mirror: tuple[Sources, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    sources, server = mirror
    assert main(["--check-urls"], sources) == 0
    assert "all URLs resolve" in capsys.readouterr().out
    (server / "dev-clean.tar.gz").unlink()
    assert main(["--check-urls"], sources) == 1
    assert "FAILED" in capsys.readouterr().out


def test_cli_full_run_prints_a_summary(
    tmp_path: Path, mirror: tuple[Sources, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    sources, _ = mirror
    code = main(["--data-dir", str(tmp_path / "d"), "--results-dir", str(tmp_path / "r")], sources)
    assert code == 0
    output = capsys.readouterr().out
    assert "DEV: 16 utterances, TEST: 24 utterances" in output
    assert "CAFE, HALL, STREET" in output
