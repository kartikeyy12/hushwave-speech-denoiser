"""Evaluation engine for M8: per-case metric rows and a deterministic parallel runner.

One *job* is one test case run through a list of systems. Jobs are independent and each case
carries its own seed, so the rows do not depend on the number of worker processes or their order.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any

from hushwave.config import DEFAULT
from hushwave.corpus import Case, realize
from hushwave.metrics import (
    delta_snr_db,
    evaluate,
    pesq_wb,
    segmental_snr_db,
    stoi_score,
)
from hushwave.systems import System, run_system

Row = dict[str, Any]
Progress = Callable[[int, int, float], None]
DELTA_COLUMNS = {
    "delta_lsd_db": ("lsd_out_db", "lsd_in_db"),
    "delta_pesq": ("pesq_out", "pesq_in"),
    "delta_stoi": ("stoi_out", "stoi_in"),
}


def _nan_safe(value: float) -> float:
    return float(value) if math.isfinite(value) else math.nan


def _external(function: Callable[[], float]) -> tuple[float, str]:
    """Run PESQ or STOI; a failure becomes NaN plus an error note instead of ending the run."""
    try:
        return _nan_safe(function()), ""
    except Exception as error:  # noqa: BLE001 - one bad file must not end a 30-minute run
        return math.nan, f"{type(error).__name__}: {error}"


def case_columns(case: Case) -> Row:
    return {
        "case": case.case_id,
        "split": case.split,
        "corpus": case.speech.corpus,
        "speech": case.speech.name,
        "speaker": case.speech.speaker,
        "noise": case.noise.name,
        "noise_kind": case.noise.kind,
        "snr_in": case.snr_db,
    }


def evaluate_case(case: Case, systems: Sequence[System]) -> list[Row]:
    """Full metric rows (one per system) for one case."""
    data = realize(case)
    x, d, y = data.clean, data.noise, data.noisy
    base = case_columns(case)
    pesq_in = stoi_in = math.nan
    notes_in = ""
    if any(s.external for s in systems):
        pesq_in, note_p = _external(lambda: pesq_wb(x, y, DEFAULT.fs))
        stoi_in, note_s = _external(lambda: stoi_score(x, y, DEFAULT.fs))
        notes_in = "; ".join(n for n in (note_p, note_s) if n)
    rows: list[Row] = []
    for system in systems:
        out = run_system(system, x, d, y)
        metrics = evaluate(x, y, out.enhanced, out.speech_component, DEFAULT, external=False)
        row: Row = {**base, "system": system.name, **metrics.as_dict()}
        note = notes_in
        if system.external:
            row["pesq_in"], row["stoi_in"] = pesq_in, stoi_in
            if system.kind == "unprocessed":
                row["pesq_out"], row["stoi_out"] = pesq_in, stoi_in
            else:
                row["pesq_out"], note_p = _external(lambda e=out.enhanced: pesq_wb(x, e))  # type: ignore[misc]
                row["stoi_out"], note_s = _external(lambda e=out.enhanced: stoi_score(x, e))  # type: ignore[misc]
                note = "; ".join(n for n in (note, note_p, note_s) if n)
        for column, (after, before) in DELTA_COLUMNS.items():
            row[column] = row[after] - row[before]
        row["external_note"] = note if system.external else ""
        rows.append(row)
    return rows


def tune_case(case: Case, systems: Sequence[System]) -> list[Row]:
    """Cheap rows for tuning: only the two SNR improvements, one STFT pass per system."""
    data = realize(case)
    rows: list[Row] = []
    for system in systems:
        z = run_system(system, data.clean, data.noise, data.noisy, need_components=False).enhanced
        rows.append(
            {
                "system": system.name,
                "case": case.case_id,
                "delta_snr_db": delta_snr_db(data.clean, data.noisy, z),
                "delta_segsnr_db": segmental_snr_db(data.clean, z)
                - segmental_snr_db(data.clean, data.noisy),
            }
        )
    return rows


def _job(args: tuple[Case, Sequence[System], str]) -> list[Row]:
    case, systems, mode = args
    return tune_case(case, systems) if mode == "tune" else evaluate_case(case, systems)


def print_progress(label: str) -> Progress:
    """A progress callback that prints a one-line status with an ETA to stderr."""
    last = [0.0]

    def report(done: int, total: int, elapsed: float) -> None:
        now = time.monotonic()
        if done != total and now - last[0] < 5.0:
            return
        last[0] = now
        eta = elapsed * (total - done) / max(done, 1)
        sys.stderr.write(
            f"\r[{label}] {done}/{total} cases, {elapsed:5.0f} s elapsed, ETA {eta:5.0f} s   "
        )
        if done == total:
            sys.stderr.write("\n")
        sys.stderr.flush()

    return report


def run_jobs(
    cases: Sequence[Case],
    systems: Sequence[System],
    mode: str = "full",
    workers: int = 1,
    progress: Progress | None = None,
) -> list[Row]:
    """Run every case through every system; rows come back in case order.

    mode "full" gives all metrics, mode "tune" only the SNR improvements. workers > 1 uses a pool
    of spawned processes (BLAS threads are pinned to 1 so workers do not fight over cores).
    """
    if mode not in ("full", "tune"):
        raise ValueError("mode must be 'full' or 'tune'")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    jobs = [(case, tuple(systems), mode) for case in cases]
    rows: list[Row] = []
    start = time.monotonic()
    if workers == 1 or len(jobs) < 2:
        for done, job in enumerate(jobs, start=1):
            rows.extend(_job(job))
            if progress is not None:
                progress(done, len(jobs), time.monotonic() - start)
        return rows
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for done, result in enumerate(pool.map(_job, jobs, chunksize=2), start=1):
            rows.extend(result)
            if progress is not None:
                progress(done, len(jobs), time.monotonic() - start)
    return rows
