"""The theory document must be generated from code and never drift."""

from __future__ import annotations

from pathlib import Path

import pytest

from hushwave.docfill import ROOT, build_values, generate, main, render


def test_theory_md_is_up_to_date() -> None:
    assert (ROOT / "docs" / "THEORY.md").read_text(encoding="utf-8") == generate()


def test_theory_md_has_no_unresolved_placeholders() -> None:
    assert "@@" not in (ROOT / "docs" / "THEORY.md").read_text(encoding="utf-8")


def test_render_substitutes_values() -> None:
    assert render("a @@x@@ b @@y@@", {"x": "1", "y": "2"}) == "a 1 b 2"


def test_render_rejects_missing_value() -> None:
    with pytest.raises(ValueError, match="no value"):
        render("@@missing@@", {})


def test_render_rejects_unused_value() -> None:
    with pytest.raises(ValueError, match="never used"):
        render("plain text", {"extra": "1"})


def test_build_values_contains_key_numbers() -> None:
    values = build_values()
    assert values["fs_hz"] == "16000"
    assert values["bin_hz"] == "31.25"
    assert values["sqnr_fs_db"] == "98.1"


def test_main_check_and_write(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    template = (ROOT / "docs" / "THEORY.template.md").read_text(encoding="utf-8")
    (tmp_path / "docs" / "THEORY.template.md").write_text(template, encoding="utf-8")
    assert main(["--check"], root=tmp_path) == 1
    assert main([], root=tmp_path) == 0
    assert main(["--check"], root=tmp_path) == 0


def test_gain_values_in_the_docs_come_from_the_code() -> None:
    values = build_values()
    assert values["ss_a"] == "2"
    assert values["g_min"] == "0.1"
    assert values["g_min_db"] == "-20"
    assert values["gain_mu"] == "0.5"
    assert values["gain_tau_ms"] == "23"
    assert values["gain_spike_pct"] == "50"
    assert values["ss_escape_pct"] == "0.72"
    assert values["ss_escape_a1_pct"] == "29.1"


def test_gain_tables_have_the_expected_rows() -> None:
    values = build_values()
    gain_rows = values["gain_table"].splitlines()
    assert len(gain_rows) == 2 + 5
    assert "5.31" in gain_rows[3]
    noise_rows = values["noise_only_table"].splitlines()
    assert len(noise_rows) == 2 + 4
    assert values["lsa_escape_pct"] != values["dd_escape_pct"]
