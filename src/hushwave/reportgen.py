"""Numbers and tables for the report (``make report``), all taken from results/results.json.

``report/report.tex`` contains the prose. Every number in it is a ``\\val...`` macro defined in
``report/generated/numbers.tex``, and every table is a file in ``report/generated``; none is
typed by hand. The prose also makes qualitative statements ("B5 is best at every input SNR").
``check_claims`` tests each of them against the results and raises ``ClaimError`` when one no
longer holds, so the text cannot silently disagree with the data.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hushwave.live import ROOT
from hushwave.theory import aliasing_experiment, quantization_sqnr_db

Json = dict[str, Any]
REPORT_DIR = ROOT / "report"
GENERATED = REPORT_DIR / "generated"
SYSTEMS = ("B1", "B2", "B3", "B4", "B5")
HOLM_ALPHA = 0.05


class ClaimError(RuntimeError):
    """A statement in the report text is not supported by the results file."""


# ------------------------------------------------------------------ formatting
def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def tn(value: Any, digits: int = 2) -> str:
    """A number for LaTeX text: a real minus sign in math mode, 'n/a' when undefined."""
    if _missing(value):
        return "n/a"
    text = f"{abs(float(value)):.{digits}f}"
    if float(text) == 0.0:
        return text
    return f"$-${text}" if float(value) < 0 else text


def ci(entry: Json | None, digits: int = 2) -> str:
    if not entry or _missing(entry.get("mean")):
        return "n/a"
    return f"{tn(entry['mean'], digits)} [{tn(entry['lo'], digits)}, {tn(entry['hi'], digits)}]"


def pval(value: Any) -> str:
    if _missing(value):
        return "n/a"
    return "$<$0.001" if float(value) < 0.001 else f"{float(value):.3f}"


def tex_text(text: str) -> str:
    """Escape the few characters of label text that LaTeX treats specially."""
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


# ---------------------------------------------------------------------- access
def _primary(results: Json) -> tuple[str, Json]:
    corpus = results["test"]["primary_corpus"]
    return corpus, results["test"]["tables"][corpus]


def _entry(table: Json, system: str, metric: str) -> Json | None:
    value = table.get(system, {}).get(metric)
    return value if isinstance(value, dict) else None


def _mean(table: Json, system: str, metric: str) -> float:
    entry = _entry(table, system, metric)
    return float("nan") if not entry or _missing(entry.get("mean")) else float(entry["mean"])


def _bound(table: Json, system: str, metric: str, key: str) -> float:
    entry = _entry(table, system, metric)
    return float("nan") if not entry or _missing(entry.get(key)) else float(entry[key])


def _test(results: Json, group: str, vs: str, metric: str) -> Json:
    corpus = results["test"]["primary_corpus"]
    for row in results["test"][group][corpus]:
        if row["vs"] == vs and row["metric"] == metric:
            found: Json = row
            return found
    raise KeyError(f"no paired test of B5 against {vs} on {metric}")


def _labels(results: Json) -> dict[str, str]:
    return {s["name"]: s["label"] for s in results["test"]["systems"]}


def _scene(results: Json, name: str) -> Json:
    scene: Json = results["real_recordings"]["scenes"][name]
    return scene


# ---------------------------------------------------------------------- claims
def check_claims(results: Json, firmware: Json | None = None) -> list[str]:
    """Every qualitative statement of report.tex, checked; returns the ones that fail."""
    _, table = _primary(results)
    overall = table["overall"]
    by_snr = table["by_snr"]
    by_noise = table["by_noise"]
    failed: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            failed.append(message)

    best = all(
        _mean(by_snr[s], "B5", "delta_snr_db")
        >= max(_mean(by_snr[s], b, "delta_snr_db") for b in ("B1", "B2", "B3", "B4"))
        for s in by_snr
    )
    require(best, "B5 has the highest SNR improvement at every input SNR")
    for metric in ("delta_snr_db", "delta_segsnr_db"):
        row = _test(results, "tests_vs_baselines", "B1", metric)
        require(
            row["mean_diff"] > 0 and row["p_utterance_level_holm"] < HOLM_ALPHA,
            f"B5 beats B1 on {metric} after Holm correction",
        )
    pesq = _test(results, "tests_vs_baselines", "B1", "delta_pesq")
    require(pesq["p_utterance_level_holm"] >= HOLM_ALPHA, "B5 and B1 do not differ on PESQ")
    stoi = _test(results, "tests_vs_baselines", "B1", "delta_stoi")
    require(
        stoi["mean_diff"] < 0 and stoi["p_utterance_level_holm"] < HOLM_ALPHA,
        "B5 is slightly below B1 on STOI",
    )
    small_stoi = all(abs(_mean(overall, s, "delta_stoi")) < 0.02 for s in ("B1", "B4", "B5"))
    require(small_stoi, "B1, B4 and B5 change STOI by less than 0.02")
    nr = {s: _mean(overall, s, "noise_reduction_db") for s in SYSTEMS}
    require(max(nr, key=lambda k: nr[k]) == "B3", "B3 has the largest noise reduction")
    require(_mean(overall, "B3", "delta_snr_db") < 0, "B3 lowers the waveform SNR")
    mnr = {s: _mean(overall, s, "musical_noise_ratio") for s in SYSTEMS}
    require(mnr["B5"] > mnr["B1"] and mnr["B5"] > mnr["B4"], "B5 has the highest MNR of B1, B4, B5")
    for name in ("speech_in_fan", "speech_in_corridor"):
        systems = _scene(results, name)["systems"]
        require(
            systems["B5"]["noise_reduction_pause_db"] < systems["B1"]["noise_reduction_pause_db"],
            f"B5 reduces pause noise less than B1 in {name}",
        )
    rise = _test(results, "tests_vs_ablations", "A:rise-off", "delta_snr_db")
    require(
        rise["p_utterance_level_holm"] >= HOLM_ALPHA, "the rise limit gives no significant dSNR"
    )
    require(
        _mean(overall, "A:rise-off", "noise_reduction_db")
        > _mean(overall, "B5", "noise_reduction_db"),
        "switching the rise limit off raises noise reduction",
    )
    transient = _mean(by_noise["syn-transient"], "B5", "delta_snr_db")
    require(-1.0 < transient < 0.5, "no useful gain on transient noise")
    require(
        _mean(by_noise["syn-babble"], "B1", "delta_snr_db")
        > _mean(by_noise["syn-babble"], "B5", "delta_snr_db"),
        "B1 beats B5 on babble",
    )
    mins = _test(results, "tests_vs_ablations", "A:est-minstats", "delta_snr_db")
    require(
        mins["mean_diff"] < 0 and mins["p_utterance_level_holm"] < HOLM_ALPHA,
        "the minimum statistics variant has a slightly higher dSNR than B5",
    )
    for variant in ("A:prefilter-biquad", "A:prefilter-none"):
        row = _test(results, "tests_vs_ablations", variant, "delta_snr_db")
        require(
            row["mean_diff"] > 0 and row["p_utterance_level_holm"] < HOLM_ALPHA,
            f"B5 beats {variant} on dSNR",
        )
    scores = results["tuning"]["scores"]
    require(scores["B5"] > scores["B1"], "B5 beats B1 on the DEV objective")
    require(results["protocol"]["test_evaluations"] == 1, "TEST was evaluated exactly once")
    require(bool(results["meta"]["complete"]), "results.json is complete")
    own = results["test"]["tables"]["own"]["overall"]
    require(
        _mean(own, "B5", "delta_snr_db") > _mean(own, "B1", "delta_snr_db"),
        "B5 is above B1 on the own-voice corpus",
    )
    if firmware is not None:
        steps = float(firmware["max_abs_difference"]) / float(firmware["pcm_step"])
        require(steps < 2.0, "the firmware agrees with the reference within two 16-bit steps")
        require(all(r["finite"] for r in firmware["signals"]), "the firmware output is finite")
        require(
            float(firmware["uplink_bytes_per_second"])
            < 0.5 * float(firmware["baud_bytes_per_second"]),
            "the serial link is loaded below half",
        )
    return failed


# --------------------------------------------------------------------- numbers
def numbers(
    results: Json, performance: Json, tuning: Json, firmware: Json | None = None
) -> dict[str, str]:
    corpus, table = _primary(results)
    by_snr, by_noise = table["by_snr"], table["by_noise"]
    test = results["test"]
    own = test["tables"]["own"]["overall"]
    fan, corridor = (
        _scene(results, "speech_in_fan")["systems"],
        _scene(results, "speech_in_corridor")["systems"],
    )
    final = tuning["final"]["b5"]
    gain, est = dict(final["gain"]), dict(final["est"])
    b1_gain = dict(tuning["final"]["b1_gain"])
    meta = results["meta"]
    snr_grid = results["protocol"]["snr_grid_db"]
    alias = aliasing_experiment()
    timing = next(r for r in performance["timing"] if r["block"] == 256)
    v: dict[str, str] = {
        "Corpus": corpus,
        "NUtt": str(table["n_utterances"]),
        "NOwn": str(test["tables"]["own"]["n_utterances"]),
        "NCases": str(table["n_cases"]),
        "NNoise": str(len(test["noises"])),
        "NSnr": str(len(test["snrs_db"])),
        "SnrGrid": ", ".join(tn(s, 0) for s in snr_grid),
        "NTune": str(results["tuning"]["n_cases"]),
        "Active": f"{100 * test['speech_activity'][corpus]['mean']:.0f}",
        "TestEvals": str(results["protocol"]["test_evaluations"]),
    }
    overall = table["overall"]
    v["DsnrBone"] = tn(_mean(overall, "B1", "delta_snr_db"))
    v["DsnrBtwo"] = tn(_mean(overall, "B2", "delta_snr_db"))
    v["DsnrBthree"] = tn(_mean(overall, "B3", "delta_snr_db"))
    v["DsnrBfour"] = tn(_mean(overall, "B4", "delta_snr_db"))
    v["DsnrBfive"] = tn(_mean(overall, "B5", "delta_snr_db"))
    v["DsnrBfiveLo"] = tn(_bound(overall, "B5", "delta_snr_db", "lo"))
    v["DsnrBfiveHi"] = tn(_bound(overall, "B5", "delta_snr_db", "hi"))
    v["SegBone"] = tn(_mean(overall, "B1", "delta_segsnr_db"))
    v["SegBfive"] = tn(_mean(overall, "B5", "delta_segsnr_db"))
    v["NrBone"] = tn(_mean(overall, "B1", "noise_reduction_db"), 1)
    v["NrBthree"] = tn(_mean(overall, "B3", "noise_reduction_db"), 1)
    v["NrBfive"] = tn(_mean(overall, "B5", "noise_reduction_db"), 1)
    v["MnrBone"] = tn(_mean(overall, "B1", "musical_noise_ratio"))
    v["MnrBfour"] = tn(_mean(overall, "B4", "musical_noise_ratio"))
    v["MnrBfive"] = tn(_mean(overall, "B5", "musical_noise_ratio"))
    v["PesqBone"] = tn(_mean(overall, "B1", "delta_pesq"))
    v["PesqBthree"] = tn(_mean(overall, "B3", "delta_pesq"))
    v["PesqBfive"] = tn(_mean(overall, "B5", "delta_pesq"))
    v["PesqBfiveCi"] = ci(_entry(overall, "B5", "delta_pesq"))
    v["StoiBthree"] = tn(_mean(overall, "B3", "delta_stoi"), 3)
    v["StoiBfive"] = tn(_mean(overall, "B5", "delta_stoi"), 3)
    v["DsnrFiveAtLow"] = tn(_mean(by_snr[f"{snr_grid[0]:g}"], "B5", "delta_snr_db"), 1)
    v["DsnrFiveAtHigh"] = tn(_mean(by_snr[f"{snr_grid[-1]:g}"], "B5", "delta_snr_db"), 1)
    v["SnrLow"] = tn(snr_grid[0], 0)
    v["SnrHigh"] = tn(snr_grid[-1], 0)
    v["MedDsnrOne"] = tn(_test(results, "tests_vs_baselines", "B1", "delta_snr_db")["median_diff"])
    v["MedSegOne"] = tn(
        _test(results, "tests_vs_baselines", "B1", "delta_segsnr_db")["median_diff"]
    )
    v["MedDsnrFour"] = tn(_test(results, "tests_vs_baselines", "B4", "delta_snr_db")["median_diff"])
    v["MedPesqFour"] = tn(_test(results, "tests_vs_baselines", "B4", "delta_pesq")["median_diff"])
    v["PPesqOne"] = pval(
        _test(results, "tests_vs_baselines", "B1", "delta_pesq")["p_utterance_level_holm"]
    )
    v["PDsnrOne"] = pval(
        _test(results, "tests_vs_baselines", "B1", "delta_snr_db")["p_utterance_level_holm"]
    )
    v["TransientFive"] = tn(_mean(by_noise["syn-transient"], "B5", "delta_snr_db"), 1)
    v["BabbleOne"] = tn(_mean(by_noise["syn-babble"], "B1", "delta_snr_db"), 1)
    v["BabbleFive"] = tn(_mean(by_noise["syn-babble"], "B5", "delta_snr_db"), 1)
    v["BiquadLoss"] = tn(
        _test(results, "tests_vs_ablations", "A:prefilter-biquad", "delta_snr_db")["mean_diff"], 1
    )
    v["NoneLoss"] = tn(
        _test(results, "tests_vs_ablations", "A:prefilter-none", "delta_snr_db")["mean_diff"], 1
    )
    v["RiseOffDsnr"] = tn(_mean(overall, "A:rise-off", "delta_snr_db"))
    v["RiseOffNr"] = tn(_mean(overall, "A:rise-off", "noise_reduction_db"), 1)
    v["RiseOffSeg"] = tn(
        _test(results, "tests_vs_ablations", "A:rise-off", "delta_segsnr_db")["median_diff"], 1
    )
    v["PRiseOff"] = pval(
        _test(results, "tests_vs_ablations", "A:rise-off", "delta_snr_db")["p_utterance_level_holm"]
    )
    v["MinstatsGain"] = tn(
        -_test(results, "tests_vs_ablations", "A:est-minstats", "delta_snr_db")["mean_diff"], 1
    )
    v["PMinstats"] = pval(
        _test(results, "tests_vs_ablations", "A:est-minstats", "delta_snr_db")[
            "p_utterance_level_holm"
        ]
    )
    v["FanNrFive"] = tn(fan["B5"]["noise_reduction_pause_db"], 1)
    v["FanNrOne"] = tn(fan["B1"]["noise_reduction_pause_db"], 1)
    v["CorNrFive"] = tn(corridor["B5"]["noise_reduction_pause_db"], 1)
    v["CorNrOne"] = tn(corridor["B1"]["noise_reduction_pause_db"], 1)
    v["FanMixNr"] = tn(_mean(by_noise["own-fan_noise"], "B5", "noise_reduction_db"), 1)
    v["CorMixNr"] = tn(_mean(by_noise["own-corridor_noise"], "B5", "noise_reduction_db"), 1)
    v["FanLevelThree"] = tn(fan["B3"]["speech_level_change_db"], 1)
    v["CorLevelThree"] = tn(corridor["B3"]["speech_level_change_db"], 1)
    v["OwnDsnrFive"] = ci(_entry(own, "B5", "delta_snr_db"))
    v["OwnDsnrOne"] = tn(_mean(own, "B1", "delta_snr_db"))
    v["JDevFive"] = tn(results["tuning"]["scores"]["B5"])
    v["JDevOne"] = tn(results["tuning"]["scores"]["B1"])
    v["JTestFive"] = tn(
        (_mean(overall, "B5", "delta_snr_db") + _mean(overall, "B5", "delta_segsnr_db")) / 2
    )
    v["JTestOne"] = tn(
        (_mean(overall, "B1", "delta_snr_db") + _mean(overall, "B1", "delta_segsnr_db")) / 2
    )
    v["GMin"] = f"{gain['g_min']:g}"
    v["GainMu"] = f"{gain['gain_mu']:g}"
    v["DdAlpha"] = f"{gain['dd_alpha']:g}"
    v["McraDelta"] = f"{est['delta']:g}"
    v["RiseLimit"] = f"{est['rise_db_per_s']:g}"
    v["OneA"] = f"{b1_gain['ss_a']:g}"
    v["OneGMin"] = f"{b1_gain['g_min']:g}"
    v["DelayMs"] = f"{performance['algorithmic_delay_ms']:.1f}"
    v["DelaySamples"] = str(performance["algorithmic_delay_samples"])
    v["BlockBudget"] = f"{timing['block_budget_ms']:.0f}"
    v["Rtf"] = f"{timing['real_time_factor']:.3f}"
    v["WorstBlock"] = f"{timing['worst_block_ms']:.2f}"
    v["HeapKb"] = f"{performance['peak_heap_bytes_10s_stream'] / 1024:.0f}"
    v["StreamDiff"] = f"{performance['streaming_vs_batch_max_abs_difference']:g}"
    v["AliasNaive"] = tn(alias.naive_alias_db, 1)
    v["AliasFiltered"] = tn(alias.filtered_alias_db, 1)
    v["AliasHz"] = f"{alias.predicted_alias_hz:.0f}"
    v["SqnrSixteen"] = f"{quantization_sqnr_db(16):.1f}"
    v["SqnrSixteenQuiet"] = f"{quantization_sqnr_db(16, 40.0):.1f}"
    v["SqnrEightQuiet"] = f"{quantization_sqnr_db(8, 40.0):.1f}"
    v["Commit"] = str(meta["git_commit"])
    v["Generated"] = str(meta["generated_utc"])[:10]
    v["PythonVersion"] = str(meta["python"])
    v["NumpyVersion"] = str(meta["packages"]["numpy"])
    v["ScipyVersion"] = str(meta["packages"]["scipy"])
    v["PesqVersion"] = str(meta["packages"]["pesq"])
    v["StoiVersion"] = str(meta["packages"]["pystoi"])
    v["NoisereduceVersion"] = str(meta["packages"]["noisereduce"])
    v["TuningHash"] = str(results["protocol"]["test_history"][0]["tuning_sha256"])[:16]
    if firmware is not None:
        step = float(firmware["pcm_step"])
        worst = float(firmware["max_abs_difference"])
        voiced = next(r for r in firmware["signals"] if r["signal"].startswith("voiced"))
        up_rate = float(firmware["uplink_bytes_per_second"])
        v["FwSignals"] = str(len(firmware["signals"]))
        v["FwBlocks"] = str(sum(int(r["blocks"]) for r in firmware["signals"]))
        v["FwMaxSteps"] = f"{worst / step:.1f}"
        v["FwMaxDb"] = tn(20 * math.log10(worst), 0)
        v["FwVoicedSnr"] = f"{float(voiced['snr_db']):.0f}"
        v["FwEngineKb"] = f"{int(firmware['engine_bytes']) / 1024:.0f}"
        v["FwUpKb"] = f"{float(firmware['uplink_bytes_per_second']) / 1000:.1f}"
        v["FwDownKb"] = f"{float(firmware['downlink_bytes_per_second']) / 1000:.1f}"
        v["FwLoad"] = f"{100 * up_rate / float(firmware['baud_bytes_per_second']):.0f}"
        v["FwBaud"] = str(int(firmware["baud"]))
        v["FwUpBytes"] = str(int(firmware["uplink_bytes_per_block"]))
        v["FwDownBytes"] = str(int(firmware["downlink_bytes_per_block"]))
    return v


def numbers_tex(values: dict[str, str]) -> str:
    lines = ["% Generated by make report from results/results.json. Do not edit."]
    for name, value in values.items():
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name {name!r} must contain letters only")
        lines.append(f"\\newcommand{{\\val{name}}}{{{value}}}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- tables
LABELS = {
    "B1": "spectral subtraction",
    "B2": "scipy Wiener",
    "B3": "noisereduce",
    "B4": "Wiener, plain SNR",
    "B5": "HushWave",
}


def _tabular(spec: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A table in the report style: light-blue header row, thin horizontal rules only."""
    head = " & ".join(f"\\textbf{{{h}}}" for h in header)
    lines = [
        f"\\begin{{tabular}}{{{spec}}}",
        "\\hline",
        f"\\rowcolor{{headfill}} {head} \\\\ \\hline",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\ \\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def table_results(results: Json) -> str:
    _, table = _primary(results)
    overall = table["overall"]
    rows = [
        [
            f"{name} {LABELS[name]}",
            ci(_entry(overall, name, "delta_snr_db")),
            tn(_mean(overall, name, "delta_segsnr_db")),
            tn(_mean(overall, name, "delta_pesq")),
            tn(_mean(overall, name, "delta_stoi"), 3),
            tn(_mean(overall, name, "noise_reduction_db"), 1),
            tn(_mean(overall, name, "musical_noise_ratio")),
        ]
        for name in SYSTEMS
    ]
    header = [
        "System",
        r"$\Delta$SNR (dB) [95\,\% CI]",
        r"$\Delta$segSNR (dB)",
        r"$\Delta$PESQ",
        r"$\Delta$STOI",
        "NR (dB)",
        "MNR",
    ]
    return _tabular("lrrrrrr", header, rows)


def table_data(manifest: Json) -> str:
    def row(label: str, items: list[Json]) -> list[str]:
        speakers = len({i["speaker_id"] for i in items})
        minutes = sum(float(i["duration_s"]) for i in items) / 60.0
        return [label, str(speakers), str(len(items)), f"{minutes:.1f}"]

    rows = [
        row("Development (tuning)", manifest["dev"]),
        row("Test (held out)", manifest["test"]),
    ]
    return _tabular("lrrr", ["Set", "Speakers", "Utterances", "Speech (min)"], rows)


def table_real(results: Json) -> str:
    rows: list[list[str]] = []
    scenes = (
        ("speech_in_fan", "Speech in fan noise"),
        ("speech_in_corridor", "Speech in corridor noise"),
    )
    for scene, title in scenes:
        systems = _scene(results, scene)["systems"]
        rows.append([r"\multicolumn{5}{l}{\emph{" + title + "}}"])
        for name in SYSTEMS:
            s = systems[name]
            rows.append(
                [
                    f"{name} {LABELS[name]}",
                    tn(s["noise_reduction_pause_db"], 1),
                    tn(s["residual_pause_dbfs"], 1),
                    tn(s["speech_level_change_db"], 1),
                    tn(s["musical_noise_ratio"], 2),
                ]
            )
    header = ["System", "NR in pauses (dB)", "Pause level (dBFS)", "Speech level (dB)", "MNR"]
    return _tabular("lrrrr", header, rows)


def table_firmware(firmware: Json) -> str:
    step = float(firmware["pcm_step"])
    rows = [
        [
            tex_text(r["signal"]),
            tn(float(r["max_abs_difference"]) / step, 2),
            tn(float(r["rms_difference"]) / step, 2),
        ]
        for r in firmware["signals"]
    ]
    header = ["Test signal (12 s each)", "Largest difference", "RMS difference"]
    return _tabular("lrr", header, rows)


def fragment_verification(path: Path) -> str:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            f"The verification step runs {int(data['tests_passed'])} automated tests with "
            f"{float(data['coverage_percent']):.1f}\\,\\% line coverage of the Python package "
            "and passes.\n"
        )
    return "The verification step runs the automated tests, the type checks and the style checks.\n"


def fragment_measured(path: Path) -> str:
    if not path.is_file():
        return (
            "The firmware measures the processing time of every block and reports it to the "
            "interface, which shows the current value while the system runs.\n"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    compute, trip = data["computeMs"], data["roundTripMs"]
    return (
        f"Measured on the board during a live session of {int(data['blocks'])} blocks, the "
        f"processing time of one block was {compute['mean']:.2f}\\,ms on average (95th "
        f"percentile {compute['p95']:.2f}\\,ms, largest {compute['max']:.2f}\\,ms) against a "
        "block period of 16\\,ms. "
        f"The round trip through the serial link averaged {trip['mean']:.1f}\\,ms "
        f"(largest {trip['max']:.1f}\\,ms). {int(data['dropped'])} blocks were dropped before "
        f"sending and {int(data['lost'])} were lost on the way.\n"
    )


def write_all(
    results_path: Path = ROOT / "results" / "results.json",
    performance_path: Path = ROOT / "results" / "engine_performance.json",
    tuning_path: Path = ROOT / "results" / "tuning.json",
    out_dir: Path = GENERATED,
    firmware_path: Path = ROOT / "results" / "firmware_check.json",
    measurements_path: Path = ROOT / "results" / "mcu_measurements.json",
    verification_path: Path = ROOT / "results" / "verification.json",
    manifest_path: Path = ROOT / "results" / "split_manifest.json",
) -> list[Path]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    if not firmware_path.is_file():
        raise FileNotFoundError(f"{firmware_path} is missing; run: make firmware-check")
    firmware = json.loads(firmware_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed = check_claims(results, firmware)
    if failed:
        raise ClaimError("report text no longer matches the results:\n- " + "\n- ".join(failed))
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "numbers.tex": numbers_tex(numbers(results, performance, tuning, firmware)),
        "tab_data.tex": table_data(manifest),
        "tab_results.tex": table_results(results),
        "tab_real.tex": table_real(results),
        "tab_firmware.tex": table_firmware(firmware),
        "verification.tex": fragment_verification(verification_path),
        "mcu_measured.tex": fragment_measured(measurements_path),
    }
    paths = []
    for name, text in files.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


# ------------------------------------------------------------------ viva notes
VIVA_TEMPLATE = ROOT / "docs" / "VIVA.template.md"
VIVA_OUTPUT = ROOT / "docs" / "VIVA.md"
_PLACEHOLDER = re.compile(r"@@([A-Za-z0-9]+)@@")


def plain(text: str) -> str:
    """LaTeX number text to plain text: the math-mode minus and less-than signs."""
    return text.replace("$-$", "-").replace("$<$", "<")


def render_plain(template: str, values: dict[str, str]) -> str:
    """Substitute every @@Name@@ with its plain-text value; an unknown name raises ValueError."""

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise ValueError(f"placeholder @@{name}@@ has no value")
        return plain(values[name])

    return _PLACEHOLDER.sub(substitute, template)


def write_viva(
    results_path: Path = ROOT / "results" / "results.json",
    performance_path: Path = ROOT / "results" / "engine_performance.json",
    tuning_path: Path = ROOT / "results" / "tuning.json",
    template: Path = VIVA_TEMPLATE,
    output: Path = VIVA_OUTPUT,
) -> Path:
    """Render the viva notes from their template; every number comes from the results."""
    results = json.loads(results_path.read_text(encoding="utf-8"))
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    failed = check_claims(
        results, json.loads((ROOT / "results" / "firmware_check.json").read_text("utf-8"))
    )
    if failed:
        raise ClaimError("viva notes no longer match the results:\n- " + "\n- ".join(failed))
    firmware_path = ROOT / "results" / "firmware_check.json"
    firmware = json.loads(firmware_path.read_text(encoding="utf-8"))
    text = render_plain(
        template.read_text(encoding="utf-8"), numbers(results, performance, tuning, firmware)
    )
    output.write_text(text, encoding="utf-8")
    return output


# ----------------------------------------------------------------------- build
def compile_pdf(report_dir: Path = REPORT_DIR) -> Path:
    """Run pdflatex twice (references and table of contents) and return the PDF path."""
    engine = shutil.which("pdflatex")
    if engine is None:
        raise FileNotFoundError(
            "pdflatex not found. Install it with: sudo apt install texlive-latex-recommended "
            "texlive-latex-extra texlive-fonts-recommended"
        )
    build = report_dir / "build"
    build.mkdir(exist_ok=True)
    command = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory=build",
        "report.tex",
    ]
    for _ in range(2):
        run = subprocess.run(command, cwd=report_dir, capture_output=True, text=True, check=False)
        if run.returncode != 0:
            tail = "\n".join(run.stdout.splitlines()[-25:])
            raise RuntimeError(f"pdflatex failed:\n{tail}")
    target = report_dir / "hushwave_report.pdf"
    shutil.copyfile(build / "report.pdf", target)
    return target
