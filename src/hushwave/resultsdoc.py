"""docs/RESULTS.md, generated from results/results.json (``make results-doc``).

Every number in the output comes from the results file; nothing is typed by hand. The generated
sentences only state facts (who is higher, whether a Holm-corrected test passes 0.05); the
interpretation belongs in the report and has to be written by the people who defend it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hushwave.pipeline import SYSTEM_ORDER
from hushwave.stats import ALPHA

Json = dict[str, Any]
COLUMNS = (
    ("delta_snr_db", "ΔSNR (dB)"),
    ("delta_segsnr_db", "ΔsegSNR (dB)"),
    ("noise_reduction_db", "NR (dB)"),
    ("speech_distortion_db", "SD (dB)"),
    ("delta_lsd_db", "ΔLSD (dB)"),
    ("musical_noise_ratio", "MNR"),
    ("delta_pesq", "ΔPESQ"),
    ("delta_stoi", "ΔSTOI"),
)
ABLATION_COLUMNS = (
    ("delta_snr_db", "ΔSNR (dB)"),
    ("delta_segsnr_db", "ΔsegSNR (dB)"),
    ("noise_reduction_db", "NR (dB)"),
)


def _f(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _cell(entry: Json | None, digits: int = 2) -> str:
    if not entry or _f(entry.get("mean")) == "n/a":
        return "n/a"
    low, high = _f(entry.get("lo"), digits), _f(entry.get("hi"), digits)
    return f"{_f(entry.get('mean'), digits)} [{low}, {high}]"


def _p(value: Any) -> str:
    if value is None:
        return "n/a"
    return "<0.001" if float(value) < 0.001 else f"{float(value):.3f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _header(results: Json) -> list[str]:
    meta = results["meta"]
    out = ["# Results (generated)", ""]
    if not meta.get("complete", False):
        missing = "; ".join(meta.get("missing", []))
        out += [f"> **INCOMPLETE DATA - a dry run, not the final result.** {missing}", ""]
    out += [
        f"Generated {meta.get('generated_utc', '?')} at commit `{meta.get('git_commit', '?')}` "
        f"(profile `{meta.get('profile', '?')}`, Python {meta.get('python', '?')}, "
        f"{meta.get('cpu_count', '?')} CPUs). Do not edit by hand: run `make results-doc`.",
        "",
    ]
    return out


def _protocol(results: Json, primary: str) -> list[str]:
    tuning, protocol = results["tuning"], results["protocol"]
    part = results["test"]["tables"][primary]
    grid = ", ".join(f"{s:g}" for s in protocol.get("snr_grid_db", []))
    b5 = tuning["final"]["b5"]
    scores = ", ".join(f"{k} {_f(v)}" for k, v in sorted(tuning["scores"].items()))
    return [
        "## 1. Protocol",
        "",
        f"* Primary corpus: `{primary}` ({part['n_utterances']} utterances, {part['n_cases']} "
        f"cases). Input SNR grid (dB): {grid}.",
        f"* Hyper-parameters were tuned on DEV only ({tuning['n_cases']} tuning cases; objective: "
        f"{tuning['objective']}).",
        f"* TEST was evaluated {protocol.get('test_evaluations', '?')} time(s).",
        f"* Unit of analysis: {protocol.get('unit_of_analysis', '?')}.",
        "",
        "Frozen settings of B5 (`results/tuning.json`):",
        "",
        f"* rule `{b5['rule']}`, tracker `{b5['estimator']}`, pre-filter `{b5['prefilter']}`, "
        f"frame {b5['frame_ms']} ms",
        f"* gain parameters: {dict(map(tuple, b5['gain']))}",
        f"* tracker parameters: {dict(map(tuple, b5['est']))}",
        "",
        f"DEV objective J per system (dB, higher is better): {scores}. "
        f"Strongest baseline on DEV: **{tuning['strongest_baseline']}**.",
        "",
    ]


def _overall(results: Json, primary: str, labels: dict[str, str]) -> list[str]:
    out: list[str] = []
    tables = results["test"]["tables"]
    for corpus in sorted(tables):
        kind = "primary" if corpus == primary else "secondary, single speaker"
        out += [
            f"## 2. Overall, corpus `{corpus}` ({kind})",
            "",
            "Mean over utterances with 95 % bootstrap interval. NR: noise reduction on noise-only "
            "blocks. SD: speech distortion. MNR: musical-noise ratio (1 = no more peaky than the "
            "input).",
            "",
        ]
        rows = []
        for name in SYSTEM_ORDER:
            row = tables[corpus]["overall"].get(name, {})
            rows.append(
                [f"**{name}** {labels.get(name, '')}"] + [_cell(row.get(m)) for m, _ in COLUMNS]
            )
        out += _table(["system", *[h for _, h in COLUMNS]], rows)
    return out


def _by_snr_and_noise(part: Json, primary: str) -> list[str]:
    out: list[str] = []
    snrs = sorted(part["by_snr"], key=float)
    for metric, title in (
        ("delta_snr_db", "ΔSNR (dB)"),
        ("delta_pesq", "ΔPESQ-WB"),
        ("delta_stoi", "ΔSTOI"),
    ):
        out += [f"## 3. {title} by input SNR (`{primary}`)", ""]
        rows = [
            [name] + [_cell(part["by_snr"][s].get(name, {}).get(metric)) for s in snrs]
            for name in SYSTEM_ORDER
        ]
        out += _table(["system", *[f"{float(s):g} dB" for s in snrs]], rows)
    noises = sorted(part["by_noise"])
    out += [f"## 4. ΔSNR (dB) by noise condition (`{primary}`)", ""]
    rows = [
        [name] + [_cell(part["by_noise"][n].get(name, {}).get("delta_snr_db"), 1) for n in noises]
        for name in SYSTEM_ORDER
    ]
    return out + _table(["system", *noises], rows)


def _ablation(test: Json, primary: str, labels: dict[str, str]) -> list[str]:
    overall = test["tables"][primary]["overall"]
    tests = {(e["vs"], e["metric"]): e for e in test["tests_vs_ablations"].get(primary, [])}
    out = [
        f"## 5. Ablation (`{primary}`)",
        "",
        "Each row changes one component of B5. Test: B5 minus the variant, Wilcoxon on "
        "per-utterance means, Holm-corrected within this table.",
        "",
    ]
    rows = [
        ["**B5** (full)"] + [_cell(overall["B5"].get(m)) for m, _ in ABLATION_COLUMNS] + ["", ""]
    ]
    for system in test["systems"]:
        name = system["name"]
        if not name.startswith("A:"):
            continue
        entry = tests.get((name, "delta_snr_db"), {})
        cells = [_cell(overall.get(name, {}).get(m)) for m, _ in ABLATION_COLUMNS]
        rows.append(
            [f"{name} {labels[name]}", *cells, _f(entry.get("median_diff")),
             _p(entry.get("p_utterance_level_holm"))]
        )  # fmt: skip
    header = [
        "variant",
        *[h for _, h in ABLATION_COLUMNS],
        "median B5 - variant (ΔSNR)",
        "p (Holm)",
    ]
    return out + _table(header, rows)


def _baseline_tests(test: Json, primary: str) -> list[str]:
    out = [
        f"## 6. B5 against each baseline (`{primary}`)",
        "",
        "Wilcoxon signed-rank on per-utterance means (primary), Holm-corrected over all tests of "
        "this table; the case-level p is optimistic because cases share utterances.",
        "",
    ]
    rows = []
    for e in test["tests_vs_baselines"].get(primary, []):
        direction = "higher is better" if e["higher_is_better"] else "lower is better"
        rows.append(
            [e["vs"], e["metric"], direction, str(e["n_utterances"]), _f(e["median_diff"]),
             _p(e["p_utterance_level"]), _p(e["p_utterance_level_holm"]), _p(e["p_case_level"])]
        )  # fmt: skip
    header = ["vs", "metric", "direction", "n utts", "median B5 - vs", "p", "p (Holm)", "p (cases)"]
    return out + _table(header, rows) + _headline(test, primary)


def _headline(test: Json, primary: str) -> list[str]:
    """Factual sentences about B5 versus the strongest DEV baseline, per metric."""
    strongest = test["strongest_baseline_on_dev"]
    overall = test["tables"][primary]["overall"]
    found = {
        e["metric"]: e for e in test["tests_vs_baselines"].get(primary, []) if e["vs"] == strongest
    }
    lines = [f"Against the strongest DEV baseline ({strongest}), `{primary}`:", ""]
    for metric, title in COLUMNS:
        if metric not in found:
            continue
        e = found[metric]
        diff = e["median_diff"] or 0.0
        better = diff > 0 if e["higher_is_better"] else diff < 0
        holm = e["p_utterance_level_holm"]
        if holm is None or holm >= ALPHA:
            verdict = "no significant difference"
        else:
            verdict = "B5 is better" if better else "B5 is worse"
        mine, theirs = _f(overall["B5"][metric]["mean"]), _f(overall[strongest][metric]["mean"])
        lines.append(
            f"* {title}: B5 {mine} vs {strongest} {theirs}; median paired difference "
            f"{_f(diff)}, Holm-corrected p {_p(holm)}: {verdict}."
        )
    return [*lines, ""]


def _real(results: Json) -> list[str]:
    scenes = results.get("real_recordings", {}).get("scenes", {})
    if not scenes:
        return []
    out = [
        "## 7. Real recordings without a clean reference",
        "",
        results["real_recordings"]["note"],
        "",
    ]
    for scene, info in scenes.items():
        out += [
            f"### {scene}",
            "",
            f"{_f(info['duration_s'], 1)} s; noise floor {_f(info['noise_floor_db'], 1)} dB; "
            f"pause blocks {info['pause_blocks']} of {info['total_blocks']}, speech blocks "
            f"{info['speech_blocks']}.",
            "",
        ]
        rows = [
            [name, _f(v["noise_reduction_pause_db"], 1), _f(v["residual_pause_dbfs"], 1),
             _f(v["speech_level_change_db"], 1), _f(v["musical_noise_ratio"])]
            for name, v in info["systems"].items()
        ]  # fmt: skip
        header = [
            "system",
            "pause NR (dB)",
            "pause level (dBFS)",
            "speech level change (dB)",
            "MNR",
        ]
        out += _table(header, rows)
    return out


def _flags(test: Json) -> list[str]:
    failures = test["external_failures"]
    out = ["## 8. Data-quality flags", "", f"* External-metric failures: {failures['count']}"]
    if failures["examples"]:
        out[-1] += f" (e.g. {failures['examples'][0]})"
    for corpus, a in test["speech_activity"].items():
        out.append(
            f"* Speech-active fraction, `{corpus}`: mean {_f(a['mean'])}, range "
            f"[{_f(a['min'])}, {_f(a['max'])}] over {a['n_utterances']} utterances"
        )
    return [*out, ""]


def render_results(results: Json) -> str:
    """The Markdown text of docs/RESULTS.md."""
    test = results["test"]
    primary = test["primary_corpus"]
    labels = {s["name"]: s["label"] for s in test["systems"]}
    part = test["tables"][primary]
    out = _header(results) + _protocol(results, primary) + _overall(results, primary, labels)
    out += _by_snr_and_noise(part, primary) + _ablation(test, primary, labels)
    out += _baseline_tests(test, primary) + _real(results) + _flags(test)
    return "\n".join(out)


def write_results_doc(results_path: Path, out_path: Path) -> Path:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_results(results) + "\n", encoding="utf-8")
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate docs/RESULTS.md from results.json")
    parser.add_argument("--results", type=Path, default=Path("results/results.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/RESULTS.md"))
    args = parser.parse_args(argv)
    if not args.results.exists():
        print(f"error: {args.results} not found; run `make experiments` first")
        return 1
    print(f"wrote {write_results_doc(args.results, args.out)}")
    return 0
