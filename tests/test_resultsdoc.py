from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakes import fake_results
from hushwave import resultsdoc


def test_render_contains_every_section_and_flags_incomplete_data() -> None:
    text = resultsdoc.render_results(fake_results())
    assert text.startswith("# Results (generated)")
    assert "INCOMPLETE DATA" in text
    for heading in ("## 1. Protocol", "## 2. Overall", "## 3. ΔSNR", "## 4.", "## 5. Ablation",
                    "## 6. B5 against", "## 7. Real recordings", "## 8. Data-quality"):  # fmt: skip
        assert heading in text
    assert "corpus `own` (secondary, single speaker)" in text
    assert "| **B5**" in text and "A:" in text
    assert "Strongest baseline on DEV: **B1**" in text
    assert "INCOMPLETE" not in resultsdoc.render_results(fake_results(complete=True))


def test_headline_states_significance_from_the_holm_corrected_p() -> None:
    text = resultsdoc.render_results(fake_results())
    assert "Against the strongest DEV baseline (B1)" in text
    assert "B5 is better" in text
    results = fake_results()
    for entry in results["test"]["tests_vs_baselines"]["libri"]:
        entry["p_utterance_level_holm"] = 0.5
    assert "no significant difference" in resultsdoc.render_results(results)
    for entry in results["test"]["tests_vs_baselines"]["libri"]:
        entry["p_utterance_level_holm"] = 0.001
        entry["median_diff"] = -1.0
    assert "B5 is worse" in resultsdoc.render_results(results)


def test_number_formatting_handles_missing_values() -> None:
    assert resultsdoc._f(None) == "n/a" and resultsdoc._f(float("nan")) == "n/a"
    assert resultsdoc._cell(None) == "n/a" and resultsdoc._cell({"mean": None}) == "n/a"
    assert resultsdoc._cell({"mean": 1.0, "lo": 0.5, "hi": 1.5}) == "1.00 [0.50, 1.50]"
    assert resultsdoc._p(None) == "n/a" and resultsdoc._p(0.0001) == "<0.001"
    assert resultsdoc._p(0.0456) == "0.046"


def test_real_section_is_optional(tmp_path: Path) -> None:
    results = fake_results()
    del results["real_recordings"]
    assert "## 7." not in resultsdoc.render_results(results)
    results["test"]["external_failures"] = {"count": 2, "examples": ["boom"]}
    assert "(e.g. boom)" in resultsdoc.render_results(results)


def test_write_and_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "results.json"
    source.write_text(json.dumps(fake_results()))
    out = tmp_path / "docs" / "RESULTS.md"
    assert resultsdoc.main(["--results", str(source), "--out", str(out)]) == 0
    assert out.read_text().startswith("# Results")
    assert resultsdoc.main(["--results", str(tmp_path / "none.json")]) == 1
    assert "make experiments" in capsys.readouterr().out
