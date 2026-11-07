"""Figures of the report, drawn from results/results.json (``make report``).

One style for all of them: Helvetica-like type that matches the report, a colour-blind-safe
palette in which HushWave (B5) is the only strong colour, light horizontal grid lines, no frame
on the top and the right, legends below the plot, and the size of the final print (16 cm wide).
Matplotlib draws; every plotted number comes from the results files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from PIL import Image  # noqa: E402

from hushwave.config import DEFAULT  # noqa: E402
from hushwave.live import ROOT  # noqa: E402
from hushwave.noise_est import make_estimator  # noqa: E402
from hushwave.stft import stft  # noqa: E402
from hushwave.synth import white_noise  # noqa: E402

FloatArray = NDArray[np.float64]
OUT_DIR = ROOT / "report" / "figures"
WIDTH = 6.3  # inches, the text width of the report
COLORS = {
    "B0": "#000000",
    "B1": "#D55E00",
    "B2": "#8A8A8A",
    "B3": "#CC79A7",
    "B4": "#009E73",
    "B5": "#0072B2",
}
NAMES = {
    "B1": "B1 spectral subtraction",
    "B2": "B2 scipy Wiener",
    "B3": "B3 noisereduce",
    "B4": "B4 Wiener, plain SNR",
    "B5": "B5 HushWave",
}
GREY = "#6b6b6b"


def _font() -> str:
    """The first Helvetica-like family that is installed, else Matplotlib's default."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for family in ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"):
        if family in installed:
            return family
    return "sans-serif"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": _font(),
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#dcdcdc",
            "grid.linewidth": 0.6,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "axes.axisbelow": True,
        }
    )


def _save(fig: Any, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def _primary(results: dict[str, Any]) -> dict[str, Any]:
    tables: dict[str, Any] = results["test"]["tables"][results["test"]["primary_corpus"]]
    return tables


def _cell(table: dict[str, Any], system: str, metric: str) -> tuple[float, float, float]:
    entry = table[system][metric]
    return float(entry["mean"]), float(entry["lo"]), float(entry["hi"])


def plot_delta_vs_snr(results: dict[str, Any], out: Path) -> Path:
    tables = _primary(results)
    snrs = sorted(tables["by_snr"], key=float)
    x = np.array([float(s) for s in snrs])
    panels = (
        ("delta_snr_db", "\u0394SNR (dB)"),
        ("delta_segsnr_db", "\u0394segSNR (dB)"),
        ("delta_pesq", "\u0394PESQ-WB (MOS)"),
        ("delta_stoi", "\u0394STOI"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.5), sharex=True, constrained_layout=True)
    for ax, (metric, label) in zip(axes.ravel(), panels, strict=True):
        for name in ("B3", "B2", "B4", "B1", "B5"):
            vals = np.array([_cell(tables["by_snr"][s], name, metric) for s in snrs])
            strong = name == "B5"
            ax.plot(
                x, vals[:, 0], color=COLORS[name], lw=2.2 if strong else 1.1, marker="o",
                ms=3.6 if strong else 2.6, label=NAMES[name], zorder=3 if strong else 2,
                alpha=1.0 if strong else 0.9,
            )  # fmt: skip
            if name in ("B5", "B1"):
                ax.fill_between(x, vals[:, 1], vals[:, 2], color=COLORS[name], alpha=0.14, lw=0)
        ax.axhline(0.0, color="#444444", lw=0.7)
        ax.set_ylabel(label)
        ax.set_xticks(x)
    for ax in axes[1]:
        ax.set_xlabel("Input SNR (dB)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [4, 3, 2, 1, 0]
    fig.legend(
        [handles[i] for i in order], [labels[i] for i in order], loc="outside lower center", ncol=3
    )
    return _save(fig, out)


def plot_by_noise(results: dict[str, Any], out: Path) -> Path:
    tables = _primary(results)
    noises = sorted(
        tables["by_noise"], key=lambda n: _cell(tables["by_noise"][n], "B5", "delta_snr_db")[0]
    )
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9), constrained_layout=True)
    y = np.arange(len(noises))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for pos, name in enumerate(noises):
        values = [_cell(tables["by_noise"][name], s, "delta_snr_db")[0] for s in ("B1", "B4", "B5")]
        ax.plot([min(values), max(values)], [pos, pos], color="#c8c8c8", lw=1.0, zorder=1)
    for system, marker, size in (("B1", "D", 22), ("B4", "s", 22), ("B5", "o", 46)):
        values = [_cell(tables["by_noise"][n], system, "delta_snr_db")[0] for n in noises]
        ax.scatter(values, y, s=size, marker=marker, color=COLORS[system], label=NAMES[system],
                   zorder=3, linewidths=0)  # fmt: skip
    ax.axvline(0.0, color="#444444", lw=0.7)
    ax.set_yticks(y, [n.replace("_", " ") for n in noises])
    ax.set_xlabel("\u0394SNR (dB), averaged over the input SNR grid")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3)
    return _save(fig, out)


def plot_ablation(results: dict[str, Any], out: Path) -> Path:
    tables = _primary(results)["overall"]
    labels = {s["name"]: s["label"] for s in results["test"]["systems"]}
    names = ["B5"] + [s["name"] for s in results["test"]["systems"] if s["name"].startswith("A:")]
    vals = np.array([_cell(tables, n, "delta_snr_db") for n in names])
    order = np.argsort(vals[:, 0])
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9), constrained_layout=True)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for pos, i in enumerate(order):
        full = names[i] == "B5"
        ax.barh(pos, vals[i, 0], height=0.62, color=COLORS["B5"] if full else "#a9c4de", zorder=2)
        ax.plot([vals[i, 1], vals[i, 2]], [pos, pos], color="#222222", lw=1.0, zorder=3)
        ax.text(
            vals[i, 2] + 0.06, pos, f"{vals[i, 0]:.2f}", va="center", fontsize=7.5, color="#222222"
        )
    ax.set_yticks(
        range(len(names)),
        ["B5 full system" if names[i] == "B5" else labels[names[i]] for i in order],
    )
    ax.set_xlim(0, float(vals[:, 2].max()) + 0.7)
    ax.set_xlabel("\u0394SNR (dB), mean with 95 % interval over utterances")
    return _save(fig, out)


def plot_pause_psd(results: dict[str, Any], out: Path) -> Path:
    scenes = results["real_recordings"]["scenes"]
    titles = {"speech_in_fan": "Fan noise", "speech_in_corridor": "Corridor noise"}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True, constrained_layout=True)
    for ax, name in zip(axes, titles, strict=True):
        psd = scenes[name]["psd"]
        f = np.array(psd["freq_hz"], dtype=float)[1:]
        series = (
            ("noise_reference_db", "noise-only recording",
             {"color": "#222222", "ls": (0, (4, 2)), "lw": 0.9}),
            ("B0_db", "pauses, unprocessed", {"color": "#8A8A8A", "lw": 1.3}),
            ("B1_db", "pauses, B1", {"color": COLORS["B1"], "lw": 1.1}),
            ("B5_db", "pauses, B5 HushWave", {"color": COLORS["B5"], "lw": 1.9}),
        )  # fmt: skip
        for key, label, kw in series:
            ax.semilogx(f, np.array(psd[key], dtype=float)[1:], label=label, **kw)
        ax.set_title(titles[name])
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(which="both", axis="y", visible=True)
        ax.set_xlim(30, 8000)
    axes[0].set_ylabel("PSD (dB, full-scale units)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    return _save(fig, out)


def plot_noise_tracking(out: Path, seed: int = 7) -> Path:
    cfg = DEFAULT
    rng = np.random.default_rng(seed)
    seconds = 20
    amp = np.ones(seconds * cfg.fs)
    amp[6 * cfg.fs : 12 * cfg.fs] = 10 ** (10 / 20)
    noise = white_noise(amp.size, rng) * amp * 0.05
    power = np.abs(stft(noise, cfg.frame_len, cfg.hop)) ** 2
    t = np.arange(power.shape[0]) * cfg.hop / cfg.fs
    truth = np.full(power.shape[0], 0.05**2 * cfg.frame_len * 0.375)
    truth = np.where((t >= 6) & (t < 12), truth * 10, truth)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.7), constrained_layout=True)
    ax.plot(t, 10 * np.log10(truth), color="#222222", ls=(0, (4, 2)), lw=1.0, label="true level")
    for label, kind, extra, colour, width in (
        ("minimum statistics", "minstats", {}, "#8A8A8A", 1.2),
        ("MCRA", "mcra", {}, COLORS["B1"], 1.2),
        ("MCRA with rise limit", "mcra", {"rise_db_per_s": 3.0}, COLORS["B5"], 1.9),
    ):
        estimator = make_estimator(kind, cfg.n_bins, cfg, **extra)
        level = np.array([np.mean(estimator.update(p)) for p in power])
        ax.plot(t, 10 * np.log10(level), color=colour, lw=width, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean noise PSD (dB)")
    ax.set_xlim(0, seconds)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=4)
    return _save(fig, out)


def crop_title(source: Path, out: Path, rows: int) -> Path:
    """Remove the title strip that the experiment script draws above the axes."""
    with Image.open(source) as image:
        width, height = image.size
        image.crop((0, rows, width, height)).save(out, optimize=True)
    return out


def make_all(
    results_path: Path = ROOT / "results" / "results.json",
    figures_dir: Path = ROOT / "results" / "figures",
    out_dir: Path = OUT_DIR,
) -> list[Path]:
    style()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    paths = [
        plot_delta_vs_snr(results, out_dir / "snr.pdf"),
        plot_by_noise(results, out_dir / "noise.pdf"),
        plot_ablation(results, out_dir / "ablation.pdf"),
        plot_pause_psd(results, out_dir / "psd.pdf"),
        plot_noise_tracking(out_dir / "tracking.pdf"),
    ]
    for name, target, rows in (
        ("fig_example_waveforms.png", "waveforms.png", 52),
        ("fig_example_spectrograms.png", "spectrograms.png", 0),
        ("fig_gain_per_band.png", "gain.png", 44),
    ):
        source = figures_dir / name
        if source.is_file():
            paths.append(crop_title(source, out_dir / target, rows))
    return paths
