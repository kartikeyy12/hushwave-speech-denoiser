"""Turn per-case rows into the tables and tests reported in results.json (THEORY 8.3).

Everything is grouped by corpus ("libri" is the primary result; "own" is one speaker and is
reported separately). Confidence intervals and tests use the utterance as the unit, see stats.py.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from hushwave.stats import (
    bootstrap_mean_ci,
    cluster_means,
    holm_adjust,
    paired_wilcoxon,
)

Row = dict[str, Any]
METRICS = (
    "delta_snr_db",
    "delta_segsnr_db",
    "noise_reduction_db",
    "speech_distortion_db",
    "delta_lsd_db",
    "musical_noise_ratio",
    "delta_pesq",
    "delta_stoi",
)
EXTERNAL = ("delta_pesq", "delta_stoi", "pesq_out", "stoi_out")
HIGHER_IS_BETTER = {
    "delta_snr_db": True,
    "delta_segsnr_db": True,
    "noise_reduction_db": True,
    "speech_distortion_db": False,
    "delta_lsd_db": False,
    "musical_noise_ratio": False,
    "delta_pesq": True,
    "delta_stoi": True,
}
TEST_METRICS = (
    "delta_snr_db",
    "delta_segsnr_db",
    "noise_reduction_db",
    "delta_lsd_db",
    "delta_pesq",
    "delta_stoi",
)
BOOT = 2000


def jsonable(value: Any) -> Any:
    """Recursively replace NaN and infinity by None so the result is strict JSON."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _utterance_means(rows: Sequence[Row], metric: str) -> tuple[list[str], np.ndarray]:
    return cluster_means([float(r[metric]) for r in rows], [str(r["speech"]) for r in rows])


def summarize(rows: Sequence[Row], metric: str, seed: int = 0) -> dict[str, Any]:
    """Mean over utterances of the per-utterance mean, with a 95 % bootstrap interval."""
    _, means = _utterance_means(rows, metric)
    ci = bootstrap_mean_ci(means, BOOT, 0.95, seed)
    finite = sum(1 for r in rows if math.isfinite(float(r[metric])))
    return {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "n_utts": ci.n, "n_cases": finite}


def _table(
    rows: Sequence[Row], systems: Sequence[str], metrics: Iterable[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    by_system: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)
    return {
        system: {m: summarize(by_system[system], m) for m in metrics}
        for system in systems
        if by_system.get(system)
    }


def aggregate(
    rows: Sequence[Row], systems: Sequence[str], metrics: Sequence[str] = METRICS
) -> dict[str, Any]:
    """{corpus: {overall, by_snr, by_noise}} tables for every system in ``systems``."""
    out: dict[str, Any] = {}
    for corpus in sorted({r["corpus"] for r in rows}):
        part = [r for r in rows if r["corpus"] == corpus]
        by_snr: dict[str, Any] = {}
        for snr in sorted({float(r["snr_in"]) for r in part}):
            by_snr[f"{snr:g}"] = _table(
                [r for r in part if float(r["snr_in"]) == snr], systems, metrics
            )
        by_noise: dict[str, Any] = {}
        for noise in sorted({r["noise"] for r in part}):
            by_noise[noise] = _table([r for r in part if r["noise"] == noise], systems, metrics)
        out[corpus] = {
            "overall": _table(part, systems, metrics),
            "by_snr": by_snr,
            "by_noise": by_noise,
            "n_utterances": len({r["speech"] for r in part}),
            "n_cases": len({r["case"] for r in part}),
        }
    return out


def _paired_values(
    rows: Sequence[Row], a: str, b: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-utterance means of a and b over the same utterances."""
    ra = [r for r in rows if r["system"] == a]
    rb = [r for r in rows if r["system"] == b]
    na, va = _utterance_means(ra, metric)
    nb, vb = _utterance_means(rb, metric)
    common = sorted(set(na) & set(nb))
    return (
        np.array([va[na.index(n)] for n in common]),
        np.array([vb[nb.index(n)] for n in common]),
    )


def paired_tests(
    rows: Sequence[Row], target: str, others: Sequence[str], metrics: Sequence[str] = TEST_METRICS
) -> dict[str, Any]:
    """Wilcoxon tests of ``target`` against each of ``others``, per corpus and metric.

    Primary: pairs are per-utterance means (n = number of utterances). Secondary: pairs are the
    individual cases, which are not independent and give optimistic p-values. Holm correction is
    applied across all primary tests of a corpus.
    """
    out: dict[str, Any] = {}
    for corpus in sorted({r["corpus"] for r in rows}):
        part = [r for r in rows if r["corpus"] == corpus]
        entries: list[dict[str, Any]] = []
        for other in others:
            for metric in metrics:
                a, b = _paired_values(part, target, other, metric)
                primary = paired_wilcoxon(a, b)
                case_a = {r["case"]: float(r[metric]) for r in part if r["system"] == target}
                case_b = {r["case"]: float(r[metric]) for r in part if r["system"] == other}
                keys = sorted(set(case_a) & set(case_b))
                secondary = paired_wilcoxon([case_a[k] for k in keys], [case_b[k] for k in keys])
                entries.append(
                    {
                        "vs": other,
                        "metric": metric,
                        "higher_is_better": HIGHER_IS_BETTER.get(metric, True),
                        "n_utterances": primary.n,
                        "median_diff": primary.median_diff,
                        "mean_diff": primary.mean_diff,
                        "p_utterance_level": primary.p_value,
                        "n_cases": secondary.n,
                        "p_case_level": secondary.p_value,
                    }
                )
        adjusted = holm_adjust([e["p_utterance_level"] for e in entries])
        for entry, p_holm in zip(entries, adjusted, strict=True):
            entry["p_utterance_level_holm"] = p_holm
        out[corpus] = entries
    return out


def activity_report(rows: Sequence[Row]) -> dict[str, Any]:
    """Fraction of speech-active blocks per utterance (sanity check of the 40 dB rule)."""
    result: dict[str, Any] = {}
    for corpus in sorted({r["corpus"] for r in rows}):
        per_utt: dict[str, float] = {}
        for row in rows:
            if row["corpus"] == corpus:
                per_utt[str(row["speech"])] = float(row["active_fraction"])
        values = np.array(list(per_utt.values()))
        result[corpus] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "n_utterances": int(values.size),
        }
    return result


def external_failures(rows: Sequence[Row]) -> dict[str, Any]:
    """How many rows carry an external-metric error note, with the first few distinct notes."""
    notes = [str(r.get("external_note", "")) for r in rows if r.get("external_note")]
    return {"count": len(notes), "examples": sorted(set(notes))[:5]}
