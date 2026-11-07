"""The demo interface files: references resolve, text is plain, JavaScript parses (M9)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "demo" / "web"
FILES = ("index.html", "style.css", "app.js", "worklets.js", "protocol.js", "engines.js")


@pytest.mark.parametrize("name", FILES)
def test_asset_exists_and_is_plain_ascii(name: str) -> None:
    text = (WEB / name).read_text(encoding="utf-8")
    assert text.strip()
    assert text.isascii(), f"{name} contains non-ASCII characters"


def test_every_selector_used_by_the_script_exists_in_the_page() -> None:
    page = (WEB / "index.html").read_text(encoding="utf-8")
    script = (WEB / "app.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([^"]+)"', page))
    used = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"', script))
    created_at_runtime: set[str] = set()
    assert used - ids - created_at_runtime == set()


def test_page_links_only_to_local_files() -> None:
    page = (WEB / "index.html").read_text(encoding="utf-8")
    assert "http://" not in page and "https://" not in page
    for target in re.findall(r'(?:src|href)="(/web/[^"]+)"', page):
        assert (WEB / target.removeprefix("/web/")).is_file()


def test_worklet_processors_match_the_names_the_page_uses() -> None:
    worklets = (WEB / "worklets.js").read_text(encoding="utf-8")
    script = (WEB / "app.js").read_text(encoding="utf-8")
    registered = set(re.findall(r'registerProcessor\("([a-z]+)"', worklets))
    used = set(re.findall(r'new AudioWorkletNode\([^,]+, "([a-z]+)"', script))
    assert registered == used == {"capture", "playback"}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("name", ["app.js", "worklets.js", "protocol.js", "engines.js"])
def test_javascript_parses(name: str) -> None:
    result = subprocess.run(
        ["node", "--check", str(WEB / name)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
