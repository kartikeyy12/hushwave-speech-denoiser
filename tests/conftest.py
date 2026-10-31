"""Shared pytest fixtures: every random test uses a fixed seed so runs are deterministic."""

from __future__ import annotations

import functools
import http.server
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SEED = 20260101


@pytest.fixture
def rng() -> np.random.Generator:
    """A freshly seeded generator; identical sequence in every test run."""
    return np.random.default_rng(SEED)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def serve() -> Iterator[Callable[[Path], str]]:
    """Serve a directory over HTTP on localhost; the callable returns the base URL."""
    servers: list[http.server.ThreadingHTTPServer] = []

    def start(directory: Path) -> str:
        handler = functools.partial(_QuietHandler, directory=str(directory))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def write_tone(path: Path, seconds: float, freq: float, fs: int = 16_000) -> None:
    t = np.arange(round(seconds * fs)) / fs
    sf.write(
        str(path), (0.1 * np.sin(2.0 * np.pi * freq * t)).astype(np.float64), fs, subtype="PCM_16"
    )


@pytest.fixture
def fake_librispeech(tmp_path: Path) -> Path:
    """A tiny LibriSpeech-shaped tree: 12 speakers (6 F, 6 M), 5 short FLAC files each."""
    root = tmp_path / "fake_src" / "LibriSpeech"
    lines = [";ID  |SEX| SUBSET           |MINUTES| NAME"]
    durations = [3.2, 3.4, 3.6, 3.8, 1.0]
    for index in range(12):
        speaker = str(100 + index)
        lines.append(
            f"{speaker} | {'F' if index % 2 == 0 else 'M'} | dev-clean | 8.0 | Name {index}"
        )
        folder = root / "dev-clean" / speaker / "1"
        folder.mkdir(parents=True)
        for number, duration in enumerate(durations):
            write_tone(folder / f"{speaker}-1-{number:04d}.flac", duration, 200.0 + 10 * index)
    (root / "SPEAKERS.TXT").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path / "fake_src"
