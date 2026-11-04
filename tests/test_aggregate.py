from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from hushwave.aggregate import (
    METRICS,
    activity_report,
    aggregate,
    external_failures,
    jsonable,
    paired_tests,
    summarize,
)


def _rows() -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for corpus in ("libri", "own"):
        for u in range(8):
            for noise in ("syn-white", "demand-X"):
                for snr in (0.0, 10.0):
                    base = float(rng.normal(3.0, 0.5))
                    for system, shift in (("B0", -base), ("B1", 0.0), ("B5", 1.0)):
                        row: dict[str, Any] = {
                            "case": f"{corpus}|u{u}|{noise}|{snr}",
                            "corpus": corpus,
                            "speech": f"{corpus}-u{u}",
                            "noise": noise,
                            "snr_in": snr,
                            "system": system,
                            "active_fraction": 0.6 + 0.01 * u,
                            "external_note": "",
                        }
                        for metric in METRICS:
                            row[metric] = base + shift + float(rng.normal(0, 0.05))
                        rows.append(row)
    rows[5]["delta_pesq"] = math.nan
    rows[7]["external_note"] = "RuntimeError: boom"
    return rows


def test_summarize_uses_utterance_means() -> None:
    rows = [r for r in _rows() if r["corpus"] == "libri" and r["system"] == "B1"]
    result = summarize(rows, "delta_snr_db")
    values = [r["delta_snr_db"] for r in rows]
    assert result["mean"] == pytest.approx(np.mean(values), abs=0.2)
    assert result["lo"] < result["mean"] < result["hi"]
    assert result["n_utts"] == 8 and result["n_cases"] == 32


def test_aggregate_builds_every_table() -> None:
    table = aggregate(_rows(), ["B0", "B1", "B5", "missing"])
    assert set(table) == {"libri", "own"}
    libri = table["libri"]
    assert set(libri["overall"]) == {"B0", "B1", "B5"}
    assert set(libri["by_snr"]) == {"0", "10"}
    assert set(libri["by_noise"]) == {"syn-white", "demand-X"}
    assert libri["n_utterances"] == 8 and libri["n_cases"] == 32
    gap = (
        libri["overall"]["B5"]["delta_snr_db"]["mean"]
        - libri["overall"]["B1"]["delta_snr_db"]["mean"]
    )
    assert gap == pytest.approx(1.0, abs=0.1)


def test_paired_tests_find_the_shift_and_apply_holm() -> None:
    result = paired_tests(_rows(), "B5", ["B1", "B0"])
    entries = result["libri"]
    assert len(entries) == 2 * 6
    first = next(e for e in entries if e["vs"] == "B1" and e["metric"] == "delta_snr_db")
    assert first["median_diff"] == pytest.approx(1.0, abs=0.1)
    assert first["n_utterances"] == 8 and first["n_cases"] == 32
    assert first["p_utterance_level"] < 0.02
    assert first["p_case_level"] < 1e-4
    for entry in entries:
        assert entry["p_utterance_level_holm"] >= entry["p_utterance_level"] - 1e-12


def test_activity_and_external_failures() -> None:
    report = activity_report(_rows())
    assert report["libri"]["n_utterances"] == 8
    assert report["libri"]["min"] == pytest.approx(0.6) and report["libri"]["max"] == pytest.approx(
        0.67
    )
    failures = external_failures(_rows())
    assert failures == {"count": 1, "examples": ["RuntimeError: boom"]}


def test_jsonable_makes_strict_json() -> None:
    data = {"a": math.nan, "b": [np.float64(1.5), math.inf, (np.int64(3),)], "c": {"d": -math.inf}}
    clean = jsonable(data)
    assert clean == {"a": None, "b": [1.5, None, [3]], "c": {"d": None}}
    assert jsonable(np.array([1.0, math.nan])) == [1.0, None]
    json.dumps(clean, allow_nan=False)
