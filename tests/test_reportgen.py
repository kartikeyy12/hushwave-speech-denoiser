"""Report numbers, tables, claim checks and the LaTeX build (M10)."""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from hushwave import reportgen
from hushwave.reportgen import (
    ClaimError,
    check_claims,
    ci,
    compile_pdf,
    numbers,
    numbers_tex,
    plain,
    pval,
    render_plain,
    tex_text,
    tn,
    write_all,
    write_viva,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
Json = dict[str, Any]


@pytest.fixture(scope="module")
def results() -> Json:
    data: Json = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def firmware() -> Json:
    data: Json = json.loads((ROOT / "results" / "firmware_check.json").read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def values(results: Json, firmware: Json) -> dict[str, str]:
    performance = json.loads((ROOT / "results" / "engine_performance.json").read_text("utf-8"))
    tuning = json.loads((ROOT / "results" / "tuning.json").read_text(encoding="utf-8"))
    return numbers(results, performance, tuning, firmware)


def test_number_formatting() -> None:
    assert tn(5.437) == "5.44"
    assert tn(-1.926) == "$-$1.93"
    assert tn(-0.001) == "0.00"
    assert tn(float("nan")) == "n/a" and tn(None) == "n/a"
    assert tn(3.14159, 1) == "3.1"
    assert ci({"mean": 5.44, "lo": 4.81, "hi": 6.01}) == "5.44 [4.81, 6.01]"
    assert ci({"mean": -1.0, "lo": -2.0, "hi": 0.5}) == "$-$1.00 [$-$2.00, 0.50]"
    assert ci(None) == "n/a" and ci({"mean": None}) == "n/a"
    assert pval(0.0004) == "$<$0.001" and pval(0.0415) == "0.042" and pval(None) == "n/a"


def test_tex_text_escapes_special_characters() -> None:
    assert tex_text("a_b & 5 % #1") == r"a\_b \& 5 \% \#1"


def test_plain_and_render_plain() -> None:
    assert plain("$-$5 and $<$0.001") == "-5 and <0.001"
    assert render_plain("x @@A@@ y @@B@@", {"A": "$-$1", "B": "2"}) == "x -1 y 2"
    with pytest.raises(ValueError, match="no value"):
        render_plain("@@Missing@@", {})


def test_numbers_tex_rejects_bad_macro_names() -> None:
    assert r"\newcommand{\valAbc}{1}" in numbers_tex({"Abc": "1"})
    with pytest.raises(ValueError, match="letters only"):
        numbers_tex({"A1": "1"})


def test_every_macro_used_in_the_report_is_defined(values: dict[str, str]) -> None:
    text = (REPORT / "report.tex").read_text(encoding="utf-8")
    used = set(re.findall(r"\\val([A-Za-z]+)", text))
    assert used and used <= set(values)


def test_every_placeholder_of_the_viva_template_is_defined(values: dict[str, str]) -> None:
    template = reportgen.VIVA_TEMPLATE.read_text(encoding="utf-8")
    used = set(re.findall(r"@@([A-Za-z0-9]+)@@", template))
    assert used and used <= set(values)


def test_every_generated_file_is_input_by_the_report() -> None:
    text = (REPORT / "report.tex").read_text(encoding="utf-8")
    inputs = set(re.findall(r"\\input\{generated/([a-z_]+)\}", text))
    expected = {p.stem for p in (REPORT / "generated").glob("*.tex")}
    assert expected == inputs


def test_claims_hold_for_the_real_results(results: Json, firmware: Json) -> None:
    assert check_claims(results, firmware) == []
    assert check_claims(results) == []


def test_claims_fail_for_a_firmware_that_disagrees(results: Json, firmware: Json) -> None:
    bad = copy.deepcopy(firmware)
    bad["max_abs_difference"] = 10 * bad["pcm_step"]
    bad["signals"][0]["finite"] = False
    bad["uplink_bytes_per_second"] = bad["baud_bytes_per_second"]
    failed = check_claims(results, bad)
    assert any("two 16-bit steps" in f for f in failed)
    assert any("finite" in f for f in failed)
    assert any("loaded below half" in f for f in failed)


def _tampered(results: Json) -> Json:
    data = copy.deepcopy(results)
    cell = data["test"]["tables"][data["test"]["primary_corpus"]]["by_snr"]["5"]["B5"][
        "delta_snr_db"
    ]
    cell["mean"] = 0.0
    data["tuning"]["scores"]["B5"] = 0.0
    data["protocol"]["test_evaluations"] = 2
    return data


def test_claims_fail_when_the_data_no_longer_supports_them(results: Json) -> None:
    failed = check_claims(_tampered(results))
    assert any("highest SNR improvement" in f for f in failed)
    assert any("DEV objective" in f for f in failed)
    assert any("exactly once" in f for f in failed)


def test_write_all_needs_the_firmware_check(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="firmware-check"):
        write_all(out_dir=tmp_path, firmware_path=tmp_path / "missing.json")


def test_write_all_refuses_when_a_claim_fails(results: Json, tmp_path: Path) -> None:
    bad = tmp_path / "results.json"
    bad.write_text(json.dumps(_tampered(results)), encoding="utf-8")
    with pytest.raises(ClaimError, match="no longer matches"):
        write_all(bad, out_dir=tmp_path / "gen")
    with pytest.raises(ClaimError, match="no longer match"):
        write_viva(bad, template=reportgen.VIVA_TEMPLATE, output=tmp_path / "viva.md")
    assert not (tmp_path / "gen").exists() and not (tmp_path / "viva.md").exists()


VOLATILE = {
    "verification.tex",
    "mcu_measured.tex",
}  # depend on the last verify run and on measurements


def test_write_all_matches_the_committed_files(tmp_path: Path) -> None:
    written = write_all(out_dir=tmp_path)
    assert {p.name for p in written} == {p.name for p in (REPORT / "generated").glob("*.tex")}
    for path in written:
        if path.name in VOLATILE:
            continue
        committed = (REPORT / "generated" / path.name).read_text(encoding="utf-8")
        assert path.read_text(encoding="utf-8") == committed, (
            f"{path.name} is stale: run make report"
        )


def test_tables_are_complete_tabulars() -> None:
    for path in (REPORT / "generated").glob("tab_*.tex"):
        text = path.read_text(encoding="utf-8")
        assert text.startswith("\\begin{tabular}") and text.rstrip().endswith("\\end{tabular}")
        assert "nan" not in text.lower()
        lines = text.splitlines()
        assert "\\rowcolor{headfill}" in lines[2]
        header_columns = lines[2].count("&")
        for line in lines[3:-1]:
            assert line.endswith("\\hline"), (path.name, line)
            if "multicolumn" not in line:
                assert line.count("&") == header_columns, (path.name, line)


def test_viva_notes_are_current(tmp_path: Path) -> None:
    out = write_viva(output=tmp_path / "VIVA.md")
    assert out.read_text(encoding="utf-8") == (ROOT / "docs" / "VIVA.md").read_text(
        encoding="utf-8"
    )
    assert "@@" not in out.read_text(encoding="utf-8")


def test_report_source_has_no_dash_punctuation_in_prose() -> None:
    for path in [REPORT / "report.tex"]:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("%") or any(
                skip in line for skip in ("\\bibitem", "\\draw", "tikz")
            ):
                continue
            assert "--" not in line, f"{path.name}:{number} contains a dash sequence"


def test_compile_needs_pdflatex(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="texlive"):
        compile_pdf(tmp_path)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_report_compiles(tmp_path: Path) -> None:
    shutil.copytree(
        REPORT, tmp_path / "report", ignore=shutil.ignore_patterns("build", "hushwave_report.pdf")
    )
    shutil.copytree(ROOT / "results" / "figures", tmp_path / "results" / "figures")
    pdf = compile_pdf(tmp_path / "report")
    assert pdf.is_file() and pdf.stat().st_size > 50_000


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_a_latex_error_is_reported(tmp_path: Path) -> None:
    (tmp_path / "report.tex").write_text(
        "\\documentclass{article}\\begin{document}\\undefined\\end{document}"
    )
    with pytest.raises(RuntimeError, match="pdflatex failed"):
        compile_pdf(tmp_path)


def test_fragments_without_and_with_measurements(tmp_path: Path) -> None:
    assert "automated tests" in reportgen.fragment_verification(tmp_path / "none.json")
    assert "reports it to the interface" in reportgen.fragment_measured(tmp_path / "none.json")
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps({"tests_passed": 123, "coverage_percent": 97.25}), "utf-8")
    assert "123 automated tests" in reportgen.fragment_verification(verification)
    assert "97.2" in reportgen.fragment_verification(verification)
    measured = tmp_path / "mcu.json"
    measured.write_text(
        json.dumps(
            {
                "blocks": 500,
                "computeMs": {"mean": 1.234, "p95": 2.0, "max": 3.5},
                "roundTripMs": {"mean": 20.0, "p95": 25.0, "max": 40.0},
                "dropped": 1,
                "lost": 2,
            }
        ),
        "utf-8",
    )
    text = reportgen.fragment_measured(measured)
    assert "500 blocks" in text and "1.23" in text and "3.50" in text
    assert "1 blocks were dropped" in text and "2 were lost" in text


def test_data_and_firmware_tables(firmware: Json) -> None:
    manifest = json.loads((ROOT / "results" / "split_manifest.json").read_text(encoding="utf-8"))
    table = reportgen.table_data(manifest)
    assert f" & {len(manifest['test'])} & " in table
    assert "Development" in table and "Test (held out)" in table
    text = reportgen.table_firmware(firmware)
    assert text.count("\\hline") == len(firmware["signals"]) + 2
    assert "silence & 0.00 & 0.00" in text


def test_firmware_numbers_are_defined_only_when_given(results: Json, firmware: Json) -> None:
    performance = json.loads((ROOT / "results" / "engine_performance.json").read_text("utf-8"))
    tuning = json.loads((ROOT / "results" / "tuning.json").read_text(encoding="utf-8"))
    without = numbers(results, performance, tuning)
    assert not any(name.startswith("Fw") for name in without)
    with_fw = numbers(results, performance, tuning, firmware)
    assert with_fw["FwSignals"] == str(len(firmware["signals"]))
    assert with_fw["FwBaud"] == "921600"
