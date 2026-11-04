"""Hyper-parameter selection on DEV only (docs/THEORY.md section 8.3).

Objective (higher is better): J = mean over DEV tuning cases of (delta SNR + delta segmental SNR)
/ 2, both in dB. It is cheap, needs no external package, and every candidate is scored on the
same cases. PESQ and STOI are deliberately not used for tuning, so they remain independent
indicators on TEST.

Procedure (each step is one parallel pass over the tuning cases):
 1. structure: gain rule x noise tracker x pre-filter x rise limiter (all else default);
 2. factorial grid over the interacting gain parameters (g_min, gain_mu, dd_alpha) on the
    structure winner (one-at-a-time sweeps missed the interaction between g_min and gain_mu);
 3. one-at-a-time sweeps of the rise limit, the tracker parameter and the alternative gain rule,
    around the factorial winner, and their combination, kept only if it beats the winner;
 4. the baselines B1, B2, B4 get a sweep of their own parameters, on the pipeline chosen for B5
    (B3 runs with the library defaults, as an off-the-shelf reference).
The first element of each grid is the default; ties go to the earlier element.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from hushwave.config import DEFAULT
from hushwave.corpus import Case
from hushwave.experiments import Progress, run_jobs
from hushwave.systems import PREFILTERS, System, b5_from

DEFAULT_TUNING_PER_CELL = 4
RULES = ("lsa", "wiener")
ESTIMATORS = ("mcra", "minstats")
RISE_RATES = (0.0, 3.0)
GAIN_GRID: dict[str, tuple[float, ...]] = {
    "g_min": (DEFAULT.g_min, 0.03, 0.05, 0.2),
    "gain_mu": (DEFAULT.gain_mu, 0.0, 0.3, 0.7),
    "dd_alpha": (DEFAULT.dd_alpha, 0.95, 0.9, 0.8),
}
RISE_GRID = (3.0, 1.5, 6.0, 12.0)
MCRA_DELTA_GRID = (DEFAULT.mcra_delta, 3.0, 8.0)
MINSTATS_ALPHA_GRID = (DEFAULT.ms_alpha_s, 0.7, 0.95)
B1_GRID = {"ss_a": (2.0, 1.0, 1.5, 3.0, 4.0), "g_min": (0.1, 0.05, 0.2)}
B4_GRID = (0.1, 0.03, 0.05, 0.2, 0.3)
B2_WINDOWS = (15, 5, 9, 31, 63)


@dataclass(frozen=True)
class TuningResult:
    b5: System
    b1_gain: tuple[tuple[str, float], ...]
    b4_gain: tuple[tuple[str, float], ...]
    b2_window: int
    strongest_baseline: str
    scores: dict[str, float]
    log: list[dict[str, Any]]
    n_cases: int


def objective_by_system(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[float, float, float]]:
    """(J, mean delta SNR, mean delta segSNR) per system name, over finite rows only."""
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        a, b = float(row["delta_snr_db"]), float(row["delta_segsnr_db"])
        if np.isfinite(a) and np.isfinite(b):
            grouped[row["system"]].append((a, b))
    out: dict[str, tuple[float, float, float]] = {}
    for name, pairs in grouped.items():
        mean_snr, mean_seg = np.asarray(pairs).mean(axis=0)
        out[name] = (0.5 * float(mean_snr + mean_seg), float(mean_snr), float(mean_seg))
    return out


def _default_gain() -> dict[str, float]:
    return {k: grid[0] for k, grid in GAIN_GRID.items()}


def _named(system: System, name: str, label: str | None = None) -> System:
    return replace(system, name=name, label=name if label is None else label)


def _score(
    systems: Sequence[System],
    cases: Sequence[Case],
    workers: int,
    progress: Progress | None,
    stage: str,
    log: list[dict[str, Any]],
) -> dict[str, float]:
    rows = run_jobs(cases, systems, "tune", workers, progress)
    table = objective_by_system(rows)
    by_name = {s.name: s for s in systems}
    for name, (j, a, b) in table.items():
        s = by_name[name]
        log.append(
            {
                "stage": stage,
                "name": name,
                "label": s.label,
                "rule": s.rule,
                "estimator": s.estimator,
                "prefilter": s.prefilter,
                "gain": dict(s.gain),
                "est": dict(s.est),
                "window": s.window,
                "J": j,
                "delta_snr_db": a,
                "delta_segsnr_db": b,
            }
        )
    return {name: v[0] for name, v in table.items()}


def _best(order: Sequence[str], scores: dict[str, float]) -> str:
    """Highest score; ties (within 1e-9) go to the earlier name, so defaults win ties."""
    best = order[0]
    for name in order[1:]:
        if scores.get(name, -np.inf) > scores.get(best, -np.inf) + 1e-9:
            best = name
    return best


def tune_b5(
    cases: Sequence[Case], workers: int, progress: Progress | None, log: list[dict[str, Any]]
) -> tuple[System, float]:
    """Steps 1 to 3 of the procedure for the full system. Returns (B5, its DEV score)."""
    structures = []
    for i, (rule, est, pre, rise) in enumerate(
        itertools.product(RULES, ESTIMATORS, PREFILTERS, RISE_RATES)
    ):
        base = System(
            f"S1-{i:02d}",
            f"{rule}/{est}/{pre}/rise={rise:g}",
            "gain",
            rule=rule,
            estimator=est,
            prefilter=pre,
        )
        structures.append(base.with_gain(**_default_gain()).with_est(rise_db_per_s=rise))
    scores = _score(structures, cases, workers, progress, "structure", log)
    winner = next(s for s in structures if s.name == _best([s.name for s in structures], scores))

    grid = [
        _named(
            winner.with_gain(**dict(zip(GAIN_GRID, values, strict=True))),
            f"S2-{i:02d}",
            ", ".join(f"{k}={v:g}" for k, v in zip(GAIN_GRID, values, strict=True)),
        )
        for i, values in enumerate(itertools.product(*GAIN_GRID.values()))
    ]
    grid_scores = _score(grid, cases, workers, progress, "gain-grid", log)
    base = next(s for s in grid if s.name == _best([s.name for s in grid], grid_scores))
    reference = grid_scores[base.name]

    sweeps: list[System] = []
    owner: dict[str, tuple[str, float, str]] = {}

    def add(param: str, value: float, target: str, system: System) -> None:
        name = f"S3-{len(sweeps):02d}"
        sweeps.append(_named(system, name, f"{param}={value:g}"))
        owner[name] = (param, value, target)

    if dict(base.est).get("rise_db_per_s", 0.0) > 0.0:
        for value in RISE_GRID:
            add("rise_db_per_s", value, "est", base.with_est(rise_db_per_s=value))
    est_key, est_grid = (
        ("delta", MCRA_DELTA_GRID) if base.estimator == "mcra" else ("alpha_s", MINSTATS_ALPHA_GRID)
    )
    for value in est_grid[1:]:
        add(est_key, value, "est", base.with_est(**{est_key: value}))
    other_rule = next((r for r in RULES if r != base.rule), None)
    if other_rule is not None:
        sweeps.append(
            _named(replace(base, rule=other_rule), f"S3-{len(sweeps):02d}", f"rule={other_rule}")
        )
        owner[sweeps[-1].name] = ("rule", 0.0, "rule")
    sweep_scores = _score(sweeps, cases, workers, progress, "sweep", log)

    chosen = base
    for param in sorted({p for p, _, _ in owner.values()}):
        names = [n for n, (p, _, _) in owner.items() if p == param]
        top = _best(names, {n: sweep_scores.get(n, -np.inf) for n in names})
        if sweep_scores.get(top, -np.inf) > reference + 1e-9:
            _, value, target = owner[top]
            if target == "rule":
                chosen = replace(chosen, rule=other_rule or chosen.rule)
            else:
                chosen = chosen.with_est(**{param: value})
    final, final_score = base, reference
    if chosen.structure() != base.structure():
        combo = _named(chosen, "S4-combined", "combined sweep winners")
        combo_score = _score([combo], cases, workers, progress, "combination", log)
        if combo_score.get("S4-combined", -np.inf) > reference + 1e-9:
            final, final_score = combo, combo_score["S4-combined"]
    return b5_from(_named(final, "B5", "HushWave (full)")), final_score


def tune_baselines(
    b5: System,
    cases: Sequence[Case],
    workers: int,
    progress: Progress | None,
    log: list[dict[str, Any]],
) -> tuple[dict[str, System], dict[str, float]]:
    """Step 4: B1, B2 and B4 on the pipeline of B5; B3 with library defaults."""
    pipeline = System("p", "p", "gain", estimator=b5.estimator, prefilter=b5.prefilter, est=b5.est)
    candidates: list[System] = []
    for a, g in itertools.product(B1_GRID["ss_a"], B1_GRID["g_min"]):
        system = replace(pipeline, rule="subtraction").with_gain(ss_a=a, g_min=g, gain_mu=0.0)
        candidates.append(_named(system, f"B1|a={a:g}|g={g:g}", "B1"))
    for g in B4_GRID:
        system = replace(pipeline, rule="ml_wiener").with_gain(g_min=g, gain_mu=0.0)
        candidates.append(_named(system, f"B4|g={g:g}", "B4"))
    for w in B2_WINDOWS:
        candidates.append(System(f"B2|w={w}", "B2", "scipy_wiener", window=w))
    candidates.append(System("B3", "B3", "noisereduce"))
    scores = _score(candidates, cases, workers, progress, "baselines", log)
    best: dict[str, System] = {}
    for tag in ("B1", "B2", "B4"):
        names = [c.name for c in candidates if c.name.startswith(tag + "|")]
        best[tag] = next(c for c in candidates if c.name == _best(names, scores))
    summary = {tag: scores[best[tag].name] for tag in best}
    summary["B3"] = scores["B3"]
    return best, summary


def run_tuning(
    cases: Sequence[Case], workers: int = 1, progress: Progress | None = None
) -> TuningResult:
    """The whole DEV procedure. ``cases`` should already be the tuning subset."""
    if not cases:
        raise ValueError("no tuning cases")
    log: list[dict[str, Any]] = []
    b5, b5_score = tune_b5(cases, workers, progress, log)
    best, summary = tune_baselines(b5, cases, workers, progress, log)
    strongest = max(summary, key=lambda tag: summary[tag])
    return TuningResult(
        b5=b5,
        b1_gain=best["B1"].gain,
        b4_gain=best["B4"].gain,
        b2_window=best["B2"].window,
        strongest_baseline=strongest,
        scores={**summary, "B5": b5_score},
        log=log,
        n_cases=len(cases),
    )
