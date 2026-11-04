"""Figures for the report and the viva (``make figures``), all drawn from results.json and data.

Matplotlib is used for drawing only; every number plotted comes from results.json, from the
project's own DSP code, or (the example signals) from the case recorded in
``protocol.figure_case``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from hushwave.config import DEFAULT  # noqa: E402
from hushwave.corpus import Case, Realized, build_cases, realize  # noqa: E402
from hushwave.enhancer import performance_report  # noqa: E402
from hushwave.gains import GainStage  # noqa: E402
from hushwave.metrics import speech_activity  # noqa: E402
from hushwave.noise_est import make_estimator  # noqa: E402
from hushwave.pipeline import SYSTEM_ORDER, Sources  # noqa: E402
from hushwave.stft import stft  # noqa: E402
from hushwave.synth import white_noise  # noqa: E402
from hushwave.systems import System, resolve, run_system, system_from_dict  # noqa: E402

FloatArray = NDArray[np.float64]
COLORS = {
    "B0": "#7f7f7f",
    "B1": "#e08a00",
    "B2": "#2ca02c",
    "B3": "#9467bd",
    "B4": "#1f77b4",
    "B5": "#d62728",
}
LABELS = {
    "B0": "B0 unprocessed",
    "B1": "B1 spectral subtraction",
    "B2": "B2 scipy wiener",
    "B3": "B3 noisereduce",
    "B4": "B4 Wiener (plain SNR)",
    "B5": "B5 HushWave",
}
DPI = 150


def _num(value: Any) -> float:
    return float("nan") if value is None else float(value)


def load_results(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _primary(results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    corpus = str(results["test"]["primary_corpus"])
    return corpus, results["test"]["tables"][corpus]


def _save(fig: Any, path: Path, tight: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _cell(table: dict[str, Any], system: str, metric: str) -> tuple[float, float, float]:
    entry = table.get(system, {}).get(metric, {})
    return _num(entry.get("mean")), _num(entry.get("lo")), _num(entry.get("hi"))


def plot_delta_vs_snr(results: dict[str, Any], out: Path) -> Path:
    """Improvement against input SNR for B1 to B5 (mean and 95 % interval over utterances)."""
    corpus, tables = _primary(results)
    snrs = sorted(tables["by_snr"], key=float)
    metrics = (
        ("delta_snr_db", "SNR improvement (dB)"),
        ("delta_segsnr_db", "segmental SNR improvement (dB)"),
        ("delta_pesq", "PESQ-WB improvement (MOS)"),
        ("delta_stoi", "STOI improvement"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    x = np.array([float(s) for s in snrs])
    for ax, (metric, label) in zip(axes.ravel(), metrics, strict=True):
        for name in SYSTEM_ORDER[1:]:
            vals = np.array([_cell(tables["by_snr"][s], name, metric) for s in snrs])
            width = 2.5 if name == "B5" else 1.2
            yerr = [vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]]
            ax.errorbar(
                x, vals[:, 0], yerr=yerr, color=COLORS[name], lw=width, marker="o", ms=4,
                capsize=2, label=LABELS[name],
            )  # fmt: skip
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("input SNR (dB)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Improvement over the noisy input vs input SNR ({corpus}, TEST)")
    return _save(fig, out)


def plot_ablation(results: dict[str, Any], out: Path) -> Path:
    """B5 and every ablation: SNR improvement with its interval."""
    corpus, tables = _primary(results)
    overall = tables["overall"]
    names = ["B5"] + [s["name"] for s in results["test"]["systems"] if s["name"].startswith("A:")]
    labels = {s["name"]: s["label"] for s in results["test"]["systems"]}
    vals = np.array([_cell(overall, n, "delta_snr_db") for n in names])
    order = np.argsort(vals[:, 0])
    fig, ax = plt.subplots(figsize=(8.5, 0.38 * len(names) + 1.5))
    y = np.arange(len(names))
    for pos, i in enumerate(order):
        colour = COLORS["B5"] if names[i] == "B5" else "#4c72b0"
        ax.barh(pos, vals[i, 0], color=colour, alpha=0.85)
        ax.errorbar(vals[i, 0], pos, xerr=[[vals[i, 0] - vals[i, 1]], [vals[i, 2] - vals[i, 0]]],
                    color="k", capsize=2, lw=1)  # fmt: skip
    ax.set_yticks(y, [labels.get(names[i], names[i]) if names[i] != "B5" else "B5 (full system)"
                      for i in order], fontsize=8)  # fmt: skip
    ax.axvline(0.0, color="k", lw=0.6)
    ax.set_xlabel("SNR improvement (dB), mean over utterances with 95 % interval")
    ax.set_title(f"Ablation: one component changed at a time ({corpus}, TEST)")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out)


def plot_by_noise(results: dict[str, Any], out: Path) -> Path:
    """SNR improvement per noise condition for B1 to B5."""
    corpus, tables = _primary(results)
    noises = sorted(tables["by_noise"])
    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(noises) + 3), 4.5))
    width = 0.8 / 5
    for k, name in enumerate(SYSTEM_ORDER[1:]):
        vals = np.array([_cell(tables["by_noise"][n], name, "delta_snr_db") for n in noises])
        pos = np.arange(len(noises)) + (k - 2) * width
        ax.bar(pos, vals[:, 0], width, color=COLORS[name], label=LABELS[name])
    ax.set_xticks(np.arange(len(noises)), noises, rotation=35, ha="right", fontsize=8)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel("SNR improvement (dB)")
    ax.set_title(f"By noise condition, averaged over input SNR ({corpus}, TEST)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def plot_pause_psd(results: dict[str, Any], out: Path) -> Path | None:
    """Welch PSD of the pauses of the real recordings before and after, and of the noise file."""
    scenes = results.get("real_recordings", {}).get("scenes", {})
    scenes = {k: v for k, v in scenes.items() if v.get("psd", {}).get("freq_hz")}
    if not scenes:
        return None
    fig, axes = plt.subplots(1, len(scenes), figsize=(6 * len(scenes), 4.2), squeeze=False)
    for ax, (name, scene) in zip(axes[0], scenes.items(), strict=True):
        psd = scene["psd"]
        f = np.array(psd["freq_hz"], dtype=float)
        for key, label, style in (
            ("noise_reference_db", "noise-only recording", {"color": "k", "ls": "--", "lw": 1}),
            ("B0_db", "pauses, unprocessed", {"color": COLORS["B0"]}),
            ("B1_db", "pauses, B1", {"color": COLORS["B1"]}),
            ("B5_db", "pauses, B5 HushWave", {"color": COLORS["B5"], "lw": 2}),
        ):
            if key in psd:
                ax.semilogx(f[1:], np.array([_num(v) for v in psd[key]])[1:], label=label, **style)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (dB re 1 / Hz, full-scale units)")
        ax.set_title(name.replace("_", " "))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    return _save(fig, out)


def plot_latency(report: dict[str, Any], out: Path) -> Path:
    """Real-time factor against block size, with the algorithmic delay in the title."""
    rows = report["timing"]
    blocks = [r["block"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(blocks, [r["real_time_factor"] for r in rows], "o-", color=COLORS["B5"])
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="real-time limit")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("block size (samples)")
    ax.set_ylabel("real-time factor (processing time / audio time)")
    ax.set_title(f"Streaming engine, delay {report['algorithmic_delay_ms']:.0f} ms "
                 "(this machine)")  # fmt: skip
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return _save(fig, out)


def plot_noise_tracking(out: Path, seed: int = 7) -> Path:
    """True noise level against the tracker estimates when the noise steps up and down."""
    cfg = DEFAULT
    rng = np.random.default_rng(seed)
    seconds = 20
    amp = np.ones(seconds * cfg.fs)
    amp[6 * cfg.fs : 12 * cfg.fs] = 10 ** (10 / 20)
    noise = white_noise(amp.size, rng) * amp * 0.05
    power = np.abs(stft(noise, cfg.frame_len, cfg.hop)) ** 2
    t = np.arange(power.shape[0]) * cfg.hop / cfg.fs
    truth = np.full(
        power.shape[0], 0.05**2 * cfg.frame_len * 0.375
    )  # window power gain of sqrt-Hann
    truth = np.where((t >= 6) & (t < 12), truth * 10, truth)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(t, 10 * np.log10(truth), "k--", lw=1.2, label="true level (step of +10 dB)")
    for label, kind, extra, colour in (
        ("minimum statistics", "minstats", {}, "#1f77b4"),
        ("MCRA", "mcra", {}, "#2ca02c"),
        ("MCRA + rise limit 3 dB/s", "mcra", {"rise_db_per_s": 3.0}, COLORS["B5"]),
    ):
        est = make_estimator(kind, cfg.n_bins, cfg, **extra)
        level = np.array([np.mean(est.update(p)) for p in power])
        ax.plot(t, 10 * np.log10(level), label=label, color=colour)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean noise PSD estimate (dB)")
    ax.set_title("Noise tracking on white noise with a 10 dB step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out)


def _gain_matrix(system: System, noisy: FloatArray) -> FloatArray:
    cfg, gain, est = resolve(system)
    power = np.abs(stft(noisy, cfg.frame_len, cfg.hop)) ** 2
    tracker = make_estimator(system.estimator, cfg.n_bins, cfg, **est)
    stage = GainStage(system.rule, cfg.n_bins, cfg, **gain)
    return np.stack([stage.process(p, tracker.update(p)) for p in power])


def plot_example(data: Realized, system: System, title: str, out_dir: Path) -> list[Path]:
    """Waveforms, spectrograms and mean gain per band for one example mixture."""
    cfg = DEFAULT
    z = run_system(system, data.clean, data.noise, data.noisy).enhanced
    t = np.arange(data.clean.size) / cfg.fs
    paths: list[Path] = []
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=True)
    for ax, (label, sig, colour) in zip(
        axes,
        (("clean speech", data.clean, "k"), ("noisy input", data.noisy, COLORS["B0"]),
         ("HushWave output", z, COLORS["B5"])),
        strict=True,
    ):  # fmt: skip
        ax.plot(t, sig, color=colour, lw=0.5)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=10)
    paths.append(_save(fig, out_dir / "fig_example_waveforms.png"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, layout="constrained")
    signals = (("clean speech", data.clean), ("noisy input", data.noisy), ("HushWave output", z))
    spectra = [(n, np.abs(stft(s, cfg.frame_len, cfg.hop)).T) for n, s in signals]
    top = max(float(np.max(m)) for _, m in spectra)
    for ax, (label, mag) in zip(axes, spectra, strict=True):
        db = 20 * np.log10(mag / top + 1e-6)
        image = ax.imshow(db, origin="lower", aspect="auto", vmin=-80, vmax=0, cmap="magma",
                          extent=(0, data.clean.size / cfg.fs, 0, cfg.fs / 2000))  # fmt: skip
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (kHz)")
    fig.colorbar(image, ax=axes, label="dB re peak", fraction=0.02)
    paths.append(_save(fig, out_dir / "fig_example_spectrograms.png", tight=False))

    gains = _gain_matrix(system, data.noisy)
    active = speech_activity(data.clean, cfg)
    count = min(gains.shape[0], active.size)
    freq = np.arange(gains.shape[1]) * cfg.bin_hz
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for label, mask, colour in (("speech-active blocks", active[:count], COLORS["B5"]),
                                ("noise-only blocks", ~active[:count], COLORS["B4"])):  # fmt: skip
        if np.any(mask):
            ax.semilogx(freq[1:], 20 * np.log10(gains[:count][mask].mean(axis=0)[1:] + 1e-6),
                        label=label, color=colour)  # fmt: skip
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("mean applied gain (dB)")
    ax.set_title("Average gain per frequency, example mixture")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    paths.append(_save(fig, out_dir / "fig_gain_per_band.png"))
    return paths


def find_figure_case(results: dict[str, Any], sources: Sources) -> Case | None:
    """Rebuild the case named in protocol.figure_case from the data on this machine."""
    wanted = str(results.get("protocol", {}).get("figure_case", ""))
    snrs = tuple(float(s) for s in results["protocol"].get("snr_grid_db", (5.0,)))
    speech = sources.speech("test")
    noises = sources.noises("test", ("white", "pink", "hum", "babble", "am", "transient"))
    for case in build_cases("test", speech, noises, snrs):
        if case.case_id == wanted:
            return case
    return None


def make_all(
    results_path: Path,
    out_dir: Path,
    sources: Sources | None = None,
    performance_path: Path | None = None,
) -> list[Path]:
    """Draw every figure that can be drawn from what is available; returns the files written."""
    results = load_results(results_path)
    written: list[Path] = [
        plot_delta_vs_snr(results, out_dir / "fig_delta_vs_snr.png"),
        plot_ablation(results, out_dir / "fig_ablation.png"),
        plot_by_noise(results, out_dir / "fig_by_noise.png"),
        plot_noise_tracking(out_dir / "fig_noise_tracking.png"),
    ]
    psd = plot_pause_psd(results, out_dir / "fig_pause_psd.png")
    if psd is not None:
        written.append(psd)
    if performance_path is not None and performance_path.exists():
        report = json.loads(performance_path.read_text(encoding="utf-8"))
    else:
        report = performance_report()
    written.append(plot_latency(report, out_dir / "fig_latency_rtf.png"))
    if sources is not None:
        case = find_figure_case(results, sources)
        if case is not None:
            b5 = system_from_dict(next(s for s in results["test"]["systems"] if s["name"] == "B5"))
            title = f"{case.speech.name} + {case.noise.name} at {case.snr_db:g} dB (TEST case)"
            written += plot_example(realize(case), b5, title, out_dir)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from hushwave.pipeline import discover

    parser = argparse.ArgumentParser(description="HushWave figures")
    parser.add_argument("--results", type=Path, default=Path("results/results.json"))
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--recordings", type=Path, default=Path("data/recordings"))
    parser.add_argument("--manifest", type=Path, default=Path("results/split_manifest.json"))
    parser.add_argument("--performance", type=Path, default=Path("results/engine_performance.json"))
    args = parser.parse_args(argv)
    if not args.results.exists():
        print(f"error: {args.results} not found; run `make experiments` first")
        return 1
    sources = discover(args.data_dir, args.recordings, args.manifest)
    files = make_all(args.results, args.out, sources, args.performance)
    for path in files:
        print(f"wrote {path}")
    return 0
