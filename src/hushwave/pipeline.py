"""The M8 pipeline: tune on DEV, evaluate once on TEST, score the real recordings.

Stages (``scripts/run_experiments.py --stage``): ``tune`` writes results/tuning.json, ``test``
needs it and writes the ``test`` section of results/results.json, ``real`` writes the
``real_recordings`` section, ``all`` runs the three in order. Each stage can be repeated safely;
only ``test`` is guarded: it refuses to run a second time on the same frozen settings unless
``--force-retest`` is given, and every TEST evaluation is logged in ``protocol.test_history``, so
the claim "tuned on DEV, evaluated once on TEST" can be read off the results file.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hushwave import aggregate as agg
from hushwave.corpus import (
    SNR_GRID,
    Case,
    NoiseItem,
    SpeechItem,
    build_cases,
    find_recordings,
    libri_items,
    load_speech,
    noise_items,
    own_voice_items,
    select_tuning_cases,
)
from hushwave.dataset import read_manifest
from hushwave.experiments import Row, print_progress, run_jobs
from hushwave.nonintrusive import evaluate_recording
from hushwave.synth import NOISE_KINDS
from hushwave.systems import (
    System,
    make_ablations,
    make_baselines,
    system_from_dict,
    system_to_dict,
)
from hushwave.tuning import DEFAULT_TUNING_PER_CELL, TuningResult, run_tuning

SCHEMA_VERSION = 2
SYSTEM_ORDER = ("B0", "B1", "B2", "B3", "B4", "B5")
REAL_SCENES = {"speech_in_fan": "fan_noise", "speech_in_corridor": "corridor_noise"}
PACKAGES = ("numpy", "scipy", "soundfile", "pesq", "pystoi", "noisereduce", "matplotlib")


class TestAlreadyRunError(RuntimeError):
    """TEST was already evaluated with these frozen settings."""

    __test__ = False  # not a pytest class


@dataclass(frozen=True)
class Profile:
    """How large an experiment is. ``full`` is the one reported; ``quick`` is a rehearsal."""

    name: str
    snrs: tuple[float, ...] = SNR_GRID
    synthetic: tuple[str, ...] = NOISE_KINDS
    tuning_per_cell: int = DEFAULT_TUNING_PER_CELL
    max_utterances: int | None = None


PROFILES = {
    "full": Profile("full"),
    "quick": Profile("quick", (0.0, 5.0, 10.0), ("white", "hum", "babble", "transient"), 2, 6),
}


@dataclass
class Sources:
    """What is actually available on this machine."""

    data_dir: Path
    manifest: dict[str, Any] | None
    recordings: dict[str, Path]
    missing: list[str] = field(default_factory=list)

    @property
    def has_libri(self) -> bool:
        return self.manifest is not None

    def speech(self, split: str, limit: int | None = None) -> list[SpeechItem]:
        items: list[SpeechItem] = []
        if self.manifest is not None:
            items.extend(libri_items(self.manifest, self.data_dir, split))
        if "speech_quiet" in self.recordings:
            items.extend(own_voice_items(self.recordings["speech_quiet"], split))
        if limit is not None:
            libri = [i for i in items if i.corpus == "libri"][:limit]
            own = [i for i in items if i.corpus == "own"][:limit]
            items = libri + own
        return items

    def noises(self, split: str, synthetic: Sequence[str]) -> list[NoiseItem]:
        return noise_items(self.manifest, self.data_dir, self.recordings, split, synthetic)


def discover(data_dir: Path, recordings_dir: Path, manifest_path: Path) -> Sources:
    """Find the manifest-listed data and the recordings; note what is missing instead of failing."""
    missing: list[str] = []
    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        candidate = read_manifest(manifest_path)
        files = [data_dir / e["path"] for s in ("dev", "test") for e in candidate[s]]
        files += [data_dir / info["path"] for info in candidate["noise"].values()]
        absent = [f for f in files if not f.exists()]
        if absent:
            missing.append(
                f"{len(absent)} data files from the manifest are missing, e.g. {absent[0]}"
            )
        else:
            manifest = candidate
    else:
        missing.append(f"{manifest_path} not found (run `make data`)")
    recordings = find_recordings(recordings_dir)
    for role in (
        "fan_noise",
        "corridor_noise",
        "speech_quiet",
        "speech_in_fan",
        "speech_in_corridor",
    ):
        if role not in recordings:
            missing.append(f"recording '{role}' not found in {recordings_dir}")
    return Sources(data_dir, manifest, recordings, missing)


# --------------------------------------------------------------------------- files
def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(agg.jsonable(data), indent=2, sort_keys=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict[str, Any]:
    """Versions and machine description, so a results file says what produced it."""
    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    commit = "unknown"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        )
        if out.returncode == 0:
            commit = out.stdout.strip()
    except OSError:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
        "git_commit": commit,
    }


# --------------------------------------------------------------------------- stages
def tuning_to_json(result: TuningResult, profile: Profile, per_cell: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": profile.name,
        "objective": "J = mean over DEV tuning cases of (delta SNR + delta segSNR) / 2, dB",
        "n_cases": result.n_cases,
        "tuning_cases_per_cell": per_cell,
        "scores": result.scores,
        "strongest_baseline": result.strongest_baseline,
        "final": {
            "b5": system_to_dict(result.b5),
            "b1_gain": [list(p) for p in result.b1_gain],
            "b4_gain": [list(p) for p in result.b4_gain],
            "b2_window": result.b2_window,
        },
        "log": result.log,
    }


def systems_from_tuning(tuning: dict[str, Any]) -> list[System]:
    """B0 to B5 followed by the ablations, rebuilt from the frozen settings."""
    final = tuning["final"]
    b5 = system_from_dict(final["b5"])
    b1 = tuple((str(k), float(v)) for k, v in final["b1_gain"])
    b4 = tuple((str(k), float(v)) for k, v in final["b4_gain"])
    return make_baselines(b5, b1, b4, int(final["b2_window"])) + make_ablations(b5)


def stage_tune(
    sources: Sources, profile: Profile, workers: int, results_dir: Path, retune: bool = False
) -> dict[str, Any]:
    path = results_dir / "tuning.json"
    if path.exists() and not retune:
        print(f"{path} exists; keeping it (use --retune to redo the DEV tuning)")
        return _read_json(path)
    speech = sources.speech("dev", profile.max_utterances)
    noises = sources.noises("dev", profile.synthetic)
    if not speech:
        raise RuntimeError("no DEV speech available: need LibriSpeech (make data) or Speech_quiet")
    cases = build_cases("dev", speech, noises, profile.snrs)
    chosen = select_tuning_cases(cases, profile.tuning_per_cell)
    print(f"tuning on DEV: {len(chosen)} cases from {len(speech)} utterances, {len(noises)} noises")
    started = time.monotonic()
    result = run_tuning(chosen, workers, print_progress("tune"))
    data = tuning_to_json(result, profile, profile.tuning_per_cell)
    data["seconds"] = time.monotonic() - started
    _write_json(path, data)
    print(
        f"B5 settings: {result.b5.rule}/{result.b5.estimator}/{result.b5.prefilter}, "
        f"gain {dict(result.b5.gain)}, tracker {dict(result.b5.est)}; wrote {path}"
    )
    return _read_json(path)


def _write_case_table(path: Path, rows: Sequence[Row]) -> None:
    columns = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    "" if isinstance(v, float) and math.isnan(v) else v
                    for v in (row[c] for c in columns)
                ]
            )


def stage_test(
    sources: Sources,
    profile: Profile,
    workers: int,
    results_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    tuning_path = results_dir / "tuning.json"
    if not tuning_path.exists():
        raise RuntimeError("no results/tuning.json: run the tune stage first")
    tuning = _read_json(tuning_path)
    digest = file_sha256(tuning_path)
    results_path = results_dir / "results.json"
    results = _read_json(results_path)
    history = results.get("protocol", {}).get("test_history", [])
    if any(h.get("tuning_sha256") == digest for h in history) and not force:
        raise TestAlreadyRunError(
            "TEST was already evaluated with these frozen settings. A second look would make "
            "TEST a tuning set; use --force-retest only for a documented reason (it is logged)."
        )
    speech = sources.speech("test", profile.max_utterances)
    noises = sources.noises("test", profile.synthetic)
    if not speech:
        raise RuntimeError("no TEST speech available")
    systems = systems_from_tuning(tuning)
    cases = build_cases("test", speech, noises, profile.snrs)
    print(f"TEST: {len(cases)} cases x {len(systems)} systems ({len(speech)} utterances)")
    started = time.monotonic()
    rows = run_jobs(cases, systems, "full", workers, print_progress("test"))
    seconds = time.monotonic() - started
    names = [s.name for s in systems]
    baselines = [n for n in SYSTEM_ORDER if n not in ("B0", "B5")]
    ablations = [n for n in names if n.startswith("A:")]
    corpora = sorted({r["corpus"] for r in rows})
    primary = "libri" if "libri" in corpora else corpora[0]
    test = {
        "primary_corpus": primary,
        "systems": [
            {
                **system_to_dict(s),
                "role": "baseline" if s.name in SYSTEM_ORDER else "ablation",
            }
            for s in systems
        ],
        "n_cases": len(cases),
        "n_utterances": len({c.speech.name for c in cases}),
        "noises": [n.name for n in noises],
        "snrs_db": list(profile.snrs),
        "tables": agg.aggregate(rows, names),
        "tests_vs_baselines": agg.paired_tests(rows, "B5", baselines),
        "tests_vs_ablations": agg.paired_tests(
            rows, "B5", ablations, ("delta_snr_db", "delta_segsnr_db", "noise_reduction_db")
        ),
        "strongest_baseline_on_dev": tuning["strongest_baseline"],
        "speech_activity": agg.activity_report(rows),
        "external_failures": agg.external_failures(rows),
        "seconds": seconds,
    }
    _write_case_table(results_dir / "test_cases.csv.gz", rows)
    results["test"] = test
    protocol = results.setdefault("protocol", {})
    history = list(history) + [
        {
            "tuning_sha256": digest,
            "profile": profile.name,
            "forced": bool(force and history),
            "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    ]
    protocol["test_history"] = history
    protocol["test_evaluations"] = len(history)
    protocol["tuned_on"] = "DEV only (results/tuning.json)"
    protocol["figure_case"] = _figure_case(cases)
    _finish(results, results_dir, sources, profile, tuning)
    return results


def _figure_case(cases: Sequence[Case]) -> str:
    """The case used for the example waveforms and spectrograms: fixed rule, recorded."""
    preferred = [c for c in cases if c.snr_db == 5.0] or list(cases)
    for noise in ("demand-PCAFETER", "own-fan_noise"):
        for case in preferred:
            if case.noise.name == noise and case.speech.corpus == "libri":
                return case.case_id
    return preferred[0].case_id


def stage_real(sources: Sources, profile: Profile, results_dir: Path) -> dict[str, Any] | None:
    tuning_path = results_dir / "tuning.json"
    if not tuning_path.exists():
        raise RuntimeError("no results/tuning.json: run the tune stage first")
    tuning = _read_json(tuning_path)
    systems = [s for s in systems_from_tuning(tuning) if s.name in SYSTEM_ORDER]
    scenes: dict[str, Any] = {}
    for role, noise_role in REAL_SCENES.items():
        if role not in sources.recordings:
            continue
        noisy = load_speech(SpeechItem("real", role, role, str(sources.recordings[role])))
        reference = None
        if noise_role in sources.recordings:
            reference = load_speech(
                SpeechItem("real", noise_role, noise_role, str(sources.recordings[noise_role]))
            )
        print(f"real recording {role}: {noisy.size / 16000:.1f} s")
        scenes[role] = evaluate_recording(
            noisy, reference, systems, save_dir=results_dir / "audio", stem=role
        )
    if not scenes:
        print("no real speech-in-noise recordings found; skipping the real stage")
        return None
    results = _read_json(results_dir / "results.json")
    results["real_recordings"] = {
        "note": "no clean reference exists: pauses are found with the matching noise-only "
        "recording as yardstick; see THEORY 8.3",
        "scenes": scenes,
    }
    _finish(results, results_dir, sources, profile, tuning)
    return results


def _finish(
    results: dict[str, Any],
    results_dir: Path,
    sources: Sources,
    profile: Profile,
    tuning: dict[str, Any],
) -> None:
    complete = sources.has_libri and not sources.missing
    results["schema_version"] = SCHEMA_VERSION
    results["meta"] = {
        **provenance(),
        "profile": profile.name,
        "complete": complete,
        "missing": sources.missing,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    results["tuning"] = {
        "objective": tuning["objective"],
        "n_cases": tuning["n_cases"],
        "scores": tuning["scores"],
        "strongest_baseline": tuning["strongest_baseline"],
        "final": tuning["final"],
        "seconds": tuning.get("seconds"),
    }
    results.setdefault("protocol", {}).update(
        {
            "snr_grid_db": list(profile.snrs),
            "unit_of_analysis": "utterance: per-utterance means, 95 % bootstrap CI over utterances",
            "test_note": "TEST utterances, speakers and noise halves are disjoint from DEV",
        }
    )
    _write_json(results_dir / "results.json", results)
    print(
        f"wrote {results_dir / 'results.json'}"
        + ("" if complete else "  [INCOMPLETE DATA: a dry run, not the final result]")
    )


# --------------------------------------------------------------------------- validation
def validate_results(results: dict[str, Any]) -> list[str]:
    """Problems with a results.json (empty list = valid). Used by scripts/verify.py."""
    problems: list[str] = []
    if results.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("meta", "protocol", "tuning", "test"):
        if not isinstance(results.get(key), dict):
            problems.append(f"missing section '{key}'")
    if problems:
        return problems
    test = results["test"]
    protocol = results["protocol"]
    if int(protocol.get("test_evaluations", 0)) < 1:
        problems.append("protocol.test_evaluations must be at least 1")
    names = {s["name"] for s in test.get("systems", [])}
    missing = [n for n in SYSTEM_ORDER if n not in names]
    if missing:
        problems.append(f"test.systems lacks {', '.join(missing)}")
    primary = test.get("primary_corpus")
    table = test.get("tables", {}).get(primary, {}).get("overall", {})
    for name in SYSTEM_ORDER:
        entry = table.get(name, {}).get("delta_snr_db", {})
        if not isinstance(entry.get("mean"), (int, float)):
            problems.append(f"no finite delta_snr_db mean for {name} in {primary}")
    for corpus, entries in test.get("tests_vs_baselines", {}).items():
        for entry in entries:
            p = entry.get("p_utterance_level")
            if p is not None and not 0.0 <= p <= 1.0:
                problems.append(f"p-value out of range in {corpus}: {p}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HushWave M8 experiments")
    parser.add_argument("--stage", choices=("tune", "test", "real", "all"), default="all")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--recordings", type=Path, default=Path("data/recordings"))
    parser.add_argument("--manifest", type=Path, default=Path("results/split_manifest.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--retune", action="store_true", help="redo DEV tuning even if it exists")
    parser.add_argument("--force-retest", action="store_true", help="evaluate TEST again (logged)")
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]
    sources = discover(args.data_dir, args.recordings, args.manifest)
    for line in sources.missing:
        print(f"NOTE: {line}", file=sys.stderr)
    if not sources.has_libri:
        print(
            "NOTE: LibriSpeech/DEMAND not usable: the results will be own-voice only and "
            "flagged incomplete",
            file=sys.stderr,
        )
    try:
        if args.stage in ("tune", "all"):
            stage_tune(sources, profile, args.workers, args.results_dir, args.retune)
        if args.stage in ("test", "all"):
            stage_test(sources, profile, args.workers, args.results_dir, args.force_retest)
        if args.stage in ("real", "all"):
            stage_real(sources, profile, args.results_dir)
    except (TestAlreadyRunError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
