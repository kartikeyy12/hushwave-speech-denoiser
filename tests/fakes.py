"""A complete, small, fake results.json (built with the real aggregation code) for tests."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from hushwave.aggregate import (
    METRICS,
    activity_report,
    aggregate,
    external_failures,
    jsonable,
    paired_tests,
)
from hushwave.pipeline import SYSTEM_ORDER
from hushwave.systems import System, b5_from, make_ablations, system_to_dict


def fake_results(figure_case: str = "", complete: bool = False) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    b5 = b5_from(
        System("x", "x", "gain", prefilter="spectral")
        .with_gain(g_min=0.1)
        .with_est(rise_db_per_s=3.0)
    )
    systems = [
        {**system_to_dict(System(n, f"label {n}", "unprocessed")), "role": "baseline"}
        for n in SYSTEM_ORDER[:-1]
    ] + [{**system_to_dict(b5), "role": "baseline"}]
    systems += [{**system_to_dict(a), "role": "ablation"} for a in make_ablations(b5)[:3]]
    names = [s["name"] for s in systems]
    rows: list[dict[str, Any]] = []
    for corpus in ("libri", "own"):
        for u in range(12):
            for noise in ("syn-white", "own-fan_noise"):
                for snr in (0.0, 10.0):
                    base = float(rng.normal(2.0, 0.3))
                    for name in names:
                        shift = {"B0": -base, "B5": 1.0}.get(name, 0.0)
                        row: dict[str, Any] = {
                            "case": f"{corpus}|{u}|{noise}|{snr}", "corpus": corpus,
                            "speech": f"{corpus}-u{u}", "noise": noise, "snr_in": snr,
                            "system": name, "active_fraction": 0.7, "external_note": "",
                        }  # fmt: skip
                        row.update({m: base + shift + float(rng.normal(0, 0.1)) for m in METRICS})
                        rows.append(row)
    freq = np.linspace(0, 8000, 257)
    psd = {
        "freq_hz": freq, "noise_reference_db": -90 + 0 * freq, "B0_db": -85 + 0 * freq,
        "B1_db": -92 + 0 * freq, "B5_db": -100 + 0 * freq,
    }  # fmt: skip
    scene = {
        "duration_s": 60.0, "noise_floor_db": -45.0, "floor_from_noise_recording": True,
        "pause_blocks": 700, "speech_blocks": 1500, "total_blocks": 3750, "psd": psd,
        "systems": {
            n: {"noise_reduction_pause_db": 8.0, "residual_pause_dbfs": -60.0,
                "speech_level_change_db": -0.5, "musical_noise_ratio": 1.1}
            for n in SYSTEM_ORDER
        },
    }  # fmt: skip
    baselines = [n for n in SYSTEM_ORDER if n not in ("B0", "B5")]
    ablations = [n for n in names if n.startswith("A:")]
    results = {
        "schema_version": 2,
        "meta": {
            "complete": complete,
            "missing": ["fake"],
            "generated_utc": "t",
            "git_commit": "abc",
            "profile": "fake",
            "python": "3.11",
            "cpu_count": 4,
        },  # fmt: skip
        "protocol": {
            "figure_case": figure_case,
            "snr_grid_db": [0.0, 10.0],
            "test_evaluations": 1,
            "unit_of_analysis": "utterance",
        },  # fmt: skip
        "tuning": {
            "objective": "J",
            "n_cases": 12,
            "strongest_baseline": "B1",
            "scores": {"B1": 1.0, "B2": 0.5, "B3": 0.2, "B4": 0.8, "B5": 1.5},
            "final": {"b5": system_to_dict(b5)},
        },  # fmt: skip
        "test": {
            "primary_corpus": "libri",
            "systems": systems,
            "tables": aggregate(rows, names),
            "tests_vs_baselines": paired_tests(rows, "B5", baselines),
            "tests_vs_ablations": paired_tests(
                rows, "B5", ablations, ("delta_snr_db", "delta_segsnr_db", "noise_reduction_db")
            ),
            "strongest_baseline_on_dev": "B1",
            "speech_activity": activity_report(rows),
            "external_failures": external_failures(rows),
        },
        "real_recordings": {"note": "no clean reference", "scenes": {"speech_in_fan": scene}},
    }
    loaded: dict[str, Any] = json.loads(json.dumps(jsonable(results)))
    return loaded
