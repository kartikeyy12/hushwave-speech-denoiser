"""Checksum-verified downloading against a local HTTP server, and safe extraction."""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

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

PAYLOAD = b"hushwave test payload " * 5000


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def test_download_verifies_and_caches(tmp_path: Path, serve: Callable[[Path], str]) -> None:
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv" / "file.bin").write_bytes(PAYLOAD)
    base = serve(tmp_path / "srv")
    calls: list[int] = []
    target = tmp_path / "out" / "file.bin"
    download(f"{base}/file.bin", target, _md5(PAYLOAD), lambda done, total: calls.append(done))
    assert target.read_bytes() == PAYLOAD
    assert calls and calls[-1] == len(PAYLOAD)
    assert not target.with_name("file.bin.part").exists()
    calls.clear()
    download(f"{base}/file.bin", target, _md5(PAYLOAD).upper(), lambda d, t: calls.append(d))
    assert calls == []


def test_checksum_mismatch_is_rejected_and_cleaned_up(
    tmp_path: Path, serve: Callable[[Path], str]
) -> None:
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv" / "file.bin").write_bytes(PAYLOAD)
    base = serve(tmp_path / "srv")
    target = tmp_path / "file.bin"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download(f"{base}/file.bin", target, "0" * 32)
    assert not target.exists()
    assert not target.with_name("file.bin.part").exists()


def test_corrupt_cached_file_is_replaced(tmp_path: Path, serve: Callable[[Path], str]) -> None:
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv" / "file.bin").write_bytes(PAYLOAD)
    base = serve(tmp_path / "srv")
    target = tmp_path / "file.bin"
    target.write_bytes(b"damaged")
    download(f"{base}/file.bin", target, _md5(PAYLOAD))
    assert target.read_bytes() == PAYLOAD


def test_missing_url_raises_a_clear_error(tmp_path: Path, serve: Callable[[Path], str]) -> None:
    base = serve(tmp_path)
    with pytest.raises(RuntimeError, match="download failed"):
        download(f"{base}/nope.bin", tmp_path / "x.bin", "0" * 32)
    with pytest.raises(RuntimeError, match="could not fetch"):
        fetch_text(f"{base}/nope.txt")


def test_check_url_reports_status_and_size(tmp_path: Path, serve: Callable[[Path], str]) -> None:
    (tmp_path / "file.bin").write_bytes(PAYLOAD)
    base = serve(tmp_path)
    found = check_url(f"{base}/file.bin")
    assert found.status == 200 and found.size == len(PAYLOAD)
    missing = check_url(f"{base}/absent.bin")
    assert missing.status == 404 and missing.size is None


def test_fetch_text_and_md5sum_parsing(tmp_path: Path, serve: Callable[[Path], str]) -> None:
    text = f"{_md5(b'a')}  a.tar.gz\n{_md5(b'b')} *b.zip\nnot a checksum line\n"
    (tmp_path / "md5sum.txt").write_text(text, encoding="utf-8")
    base = serve(tmp_path)
    sums = parse_md5sum(fetch_text(f"{base}/md5sum.txt"))
    assert sums == {"a.tar.gz": _md5(b"a"), "b.zip": _md5(b"b")}


def test_file_digests(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    assert md5_file(path) == "900150983cd24fb0d6963f7d28e17f72"
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def _make_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as handle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))


def test_extract_tar_extracts_files(tmp_path: Path) -> None:
    _make_tar(tmp_path / "a.tar.gz", {"dir/file.txt": b"hello"})
    extract_tar(tmp_path / "a.tar.gz", tmp_path / "out")
    assert (tmp_path / "out" / "dir" / "file.txt").read_bytes() == b"hello"


def test_extract_tar_blocks_path_traversal(tmp_path: Path) -> None:
    _make_tar(tmp_path / "evil.tar.gz", {"../escaped.txt": b"x"})
    with pytest.raises(tarfile.TarError):
        extract_tar(tmp_path / "evil.tar.gz", tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_extract_zip_member(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as bundle:
        bundle.writestr("STREET/ch01.wav", b"one")
        bundle.writestr("STREET/ch02.wav", b"two")
    name = extract_zip_member(tmp_path / "a.zip", "ch01.wav", tmp_path / "o" / "STREET.wav")
    assert name == "STREET/ch01.wav"
    assert (tmp_path / "o" / "STREET.wav").read_bytes() == b"one"
    with pytest.raises(ValueError, match="expected one member"):
        extract_zip_member(tmp_path / "a.zip", "ch0", tmp_path / "o" / "x.wav")
    with pytest.raises(ValueError, match="expected one member"):
        extract_zip_member(tmp_path / "a.zip", "missing.wav", tmp_path / "o" / "y.wav")
