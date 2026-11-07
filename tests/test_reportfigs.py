"""The report figures are drawn from the results and have the report's style."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from hushwave import reportfigs
from hushwave.live import ROOT


@pytest.fixture(scope="module")
def drawn(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("figs")
    return {p.name: p for p in reportfigs.make_all(out_dir=out)}


def test_all_figures_are_drawn(drawn: dict[str, Path]) -> None:
    assert {"snr.pdf", "noise.pdf", "ablation.pdf", "psd.pdf", "tracking.pdf"} <= set(drawn)
    for name, path in drawn.items():
        assert path.stat().st_size > 5_000, name
        if path.suffix == ".pdf":
            assert path.read_bytes()[:5] == b"%PDF-"


def test_the_committed_figures_exist() -> None:
    for name in (
        "snr.pdf",
        "noise.pdf",
        "ablation.pdf",
        "psd.pdf",
        "tracking.pdf",
        "waveforms.png",
    ):
        assert (ROOT / "report" / "figures" / name).is_file(), name


def test_style_is_applied_and_fonts_fall_back() -> None:
    reportfigs.style()
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["font.family"][0] == reportfigs._font()
    assert isinstance(reportfigs._font(), str)


def test_crop_title_removes_rows(tmp_path: Path) -> None:
    source = tmp_path / "a.png"
    Image.new("RGB", (100, 80), "white").save(source)
    out = reportfigs.crop_title(source, tmp_path / "b.png", 30)
    assert Image.open(out).size == (100, 50)


def test_missing_source_images_are_skipped(tmp_path: Path) -> None:
    paths = reportfigs.make_all(figures_dir=tmp_path / "none", out_dir=tmp_path / "out")
    assert {p.suffix for p in paths} == {".pdf"}


def test_palette_has_one_colour_per_system() -> None:
    assert set(reportfigs.COLORS) == {"B0", "B1", "B2", "B3", "B4", "B5"}
    assert len(set(reportfigs.COLORS.values())) == 6


def test_the_figures_read_only_the_results_file(tmp_path: Path) -> None:
    results = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    reportfigs.style()
    out = reportfigs.plot_by_noise(results, tmp_path / "n.pdf")
    assert out.is_file()
