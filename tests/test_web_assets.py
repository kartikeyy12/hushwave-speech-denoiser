"""The demo interface files: references resolve, text is plain, JavaScript parses (M9)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "demo" / "web"
SCRIPTS = ("app.js", "recorder.js")
FILES = (
    "index.html",
    "style.css",
    "app.js",
    "worklets.js",
    "protocol.js",
    "engines.js",
    "analysis.js",
    "charts.js",
    "recorder.js",
)


@pytest.mark.parametrize("name", FILES)
def test_asset_exists_and_is_plain_ascii(name: str) -> None:
    text = (WEB / name).read_text(encoding="utf-8")
    assert text.strip()
    assert text.isascii(), f"{name} contains non-ASCII characters"


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_selector_used_by_the_script_exists_in_the_page(name: str) -> None:
    page = (WEB / "index.html").read_text(encoding="utf-8")
    script = (WEB / name).read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([^"]+)"', page))
    used = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"', script))
    used |= set(re.findall(r'register\("([A-Za-z0-9_-]+)"', script))
    created_at_runtime: set[str] = set()
    assert used and used - ids - created_at_runtime == set()


def test_every_chart_has_a_canvas_and_a_height() -> None:
    page = (WEB / "index.html").read_text(encoding="utf-8")
    for match in re.finditer(
        r'<div id="(chart-[a-z-]+)" class="chart[^"]*" data-h="(\d+)">\s*<canvas', page
    ):
        assert int(match.group(2)) >= 150
    assert len(re.findall(r'id="chart-', page)) == 6


def test_page_links_only_to_local_files() -> None:
    page = (WEB / "index.html").read_text(encoding="utf-8")
    assert "http://" not in page and "https://" not in page
    for target in re.findall(r'(?:src|href)="(/web/[^"]+)"', page):
        assert (WEB / target.removeprefix("/web/")).is_file()


def test_worklet_processors_match_the_names_the_page_uses() -> None:
    worklets = (WEB / "worklets.js").read_text(encoding="utf-8")
    scripts = "".join((WEB / name).read_text(encoding="utf-8") for name in SCRIPTS)
    registered = set(re.findall(r'registerProcessor\("([a-z]+)"', worklets))
    used = set(re.findall(r'new AudioWorkletNode\([^,]+, "([a-z]+)"', scripts))
    assert registered == used == {"capture", "playback"}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("name", [f for f in FILES if f.endswith(".js")])
def test_javascript_parses(name: str) -> None:
    result = subprocess.run(
        ["node", "--check", str(WEB / name)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
