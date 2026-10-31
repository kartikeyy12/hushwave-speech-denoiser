"""Checksum-verified downloading and safe archive extraction (standard library only)."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

USER_AGENT = "hushwave-data/1.0 (university DSP project)"
CHUNK = 1 << 20


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    return _digest(path, "md5")


def sha256_file(path: Path) -> str:
    return _digest(path, "sha256")


def parse_md5sum(text: str) -> dict[str, str]:
    """Parse `md5sum`-style text: `<hash>  <filename>` (a leading `*` on the name is allowed)."""
    sums: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 32:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return sums


@dataclass(frozen=True)
class UrlInfo:
    url: str
    status: int
    size: int | None


def _request(
    url: str, method: str = "GET", headers: dict[str, str] | None = None
) -> urllib.request.Request:
    merged = {"User-Agent": USER_AGENT}
    merged.update(headers or {})
    return urllib.request.Request(url, method=method, headers=merged)


def check_url(url: str, timeout: float = 30.0) -> UrlInfo:
    """Confirm a URL resolves. Tries HEAD, falls back to a 1-byte ranged GET."""
    try:
        with urllib.request.urlopen(_request(url, "HEAD"), timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return UrlInfo(url, int(response.status), int(length) if length else None)
    except urllib.error.HTTPError as head_error:
        if head_error.code not in (403, 405, 501):
            return UrlInfo(url, head_error.code, None)
    try:
        request = _request(url, "GET", {"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_range = response.headers.get("Content-Range", "")
            size = int(content_range.rsplit("/", 1)[-1]) if "/" in content_range else None
            return UrlInfo(url, int(response.status), size)
    except urllib.error.HTTPError as error:
        return UrlInfo(url, error.code, None)


def fetch_text(url: str, timeout: float = 60.0) -> str:
    try:
        with urllib.request.urlopen(_request(url), timeout=timeout) as response:
            return str(response.read().decode("utf-8", errors="replace"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"could not fetch {url}: {error}") from error


def download(
    url: str,
    destination: Path,
    expected_md5: str,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download to `destination`, verifying the MD5. An existing verified file is reused."""
    expected = expected_md5.lower()
    if destination.exists() and md5_file(destination) == expected:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.md5()
    done = 0
    try:
        with (
            urllib.request.urlopen(_request(url), timeout=60.0) as response,
            partial.open("wb") as out,
        ):
            length = response.headers.get("Content-Length")
            total = int(length) if length else None
            for block in iter(lambda: response.read(CHUNK), b""):
                out.write(block)
                digest.update(block)
                done += len(block)
                if progress is not None:
                    progress(done, total)
    except urllib.error.URLError as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"download failed for {url}: {error}") from error
    if digest.hexdigest() != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {url}: expected md5 {expected}, got {digest.hexdigest()}"
        )
    partial.replace(destination)
    return destination


def extract_tar(archive: Path, destination: Path) -> None:
    """Extract a .tar.gz with Python's 'data' safety filter (no absolute paths, no escapes)."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(destination, filter="data")


def extract_zip_member(archive: Path, member_suffix: str, destination_file: Path) -> str:
    """Copy the one zip member whose name ends with `member_suffix` to `destination_file`."""
    with zipfile.ZipFile(archive) as bundle:
        matches = [n for n in bundle.namelist() if n.lower().endswith(member_suffix.lower())]
        if len(matches) != 1:
            raise ValueError(
                f"{archive.name}: expected one member ending in {member_suffix!r}, found {matches}"
            )
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open(matches[0]) as source, destination_file.open("wb") as target:
            shutil.copyfileobj(source, target)
    return matches[0]
