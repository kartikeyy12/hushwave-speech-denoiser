"""Fill docs/THEORY.template.md from code so that no number in the docs is typed by hand.

Placeholders look like @@name@@. Rendering fails loudly if a placeholder has no value or a
value is never used, so the committed docs/THEORY.md can never drift from the code.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from scipy.special import exp1

from hushwave.config import DEFAULT, Config
from hushwave.noise_est import (
    KINDS,
    McraEstimator,
    MinStatsEstimator,
    calibrated_bias_curve,
    measure_step_delay,
)
from hushwave.theory import (
    aliasing_experiment,
    butterworth_highpass_gain_db,
    quantization_sqnr_db,
)
from hushwave.windows import hann_periodic, sqrt_hann, window_metrics

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path("docs") / "THEORY.template.md"
OUTPUT = Path("docs") / "THEORY.md"
PLACEHOLDER = re.compile(r"@@([a-z0-9_]+)@@")


def _sqnr_table() -> str:
    bit_depths = (8, 12, 16)
    levels = (0.0, 20.0, 40.0, 60.0)
    lines = [
        "| Level below a full-scale sine (dB) | "
        + " | ".join(f"{b}-bit" for b in bit_depths)
        + " |",
        "|---|" + "---|" * len(bit_depths),
    ]
    for level in levels:
        cells = " | ".join(f"{quantization_sqnr_db(b, level):.1f} dB" for b in bit_depths)
        lines.append(f"| {level:.0f} | {cells} |")
    return "\n".join(lines)


def _frame_ablation_table(base: Config) -> str:
    lines = [
        "| Frame (ms) | Samples N | Hop H | Bin spacing (Hz) | Bins | Max latency (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for ms in (16, 32, 64):
        n = ms * base.fs // 1000
        cfg = Config(fs=base.fs, frame_len=n, hop=n // 2)
        lines.append(
            f"| {cfg.frame_ms:.0f} | {n} | {cfg.hop} | {cfg.bin_hz:.3f} | {cfg.n_bins} | "
            f"{1000.0 * cfg.latency_max_samples / cfg.fs:.1f} |"
        )
    return "\n".join(lines)


def _step_table(cfg: Config) -> tuple[str, float]:
    labels = {"minstats": "Minimum statistics", "mcra": "MCRA"}
    steps = (6.0, 10.0, -6.0)
    lines = [
        "| Estimator | +6 dB | +10 dB | -6 dB |",
        "|---|---|---|---|",
    ]
    down_ms = 0.0
    for kind in KINDS:
        cells = []
        for step in steps:
            frames = measure_step_delay(kind, step, n_bins=cfg.n_bins).delay_frames
            cells.append(f"{frames} frames ({frames * cfg.hop_ms:.0f} ms)")
            if step < 0:
                down_ms = max(down_ms, frames * cfg.hop_ms)
        lines.append(f"| {labels[kind]} | " + " | ".join(cells) + " |")
    return "\n".join(lines), down_ms


def build_values(cfg: Config = DEFAULT) -> dict[str, str]:
    """Every number that appears in THEORY.md, as a display string."""
    sine = window_metrics(sqrt_hann(cfg.frame_len))
    hann = window_metrics(hann_periodic(cfg.frame_len))
    alias = aliasing_experiment()

    def hp_gain(freq_hz: float) -> float:
        return butterworth_highpass_gain_db(freq_hz, cfg.hp_cutoff_hz, cfg.fs)

    xi_example, gamma_example = 1.0, 0.01
    ms_estimator = MinStatsEstimator(cfg.n_bins)
    mcra_estimator = McraEstimator(cfg.n_bins)
    curve = calibrated_bias_curve(
        cfg.n_bins, alpha_s=cfg.ms_alpha_s, n_sub=cfg.ms_n_sub, sub_frames=cfg.ms_sub_frames
    )
    step_table, step_down_ms = _step_table(cfg)
    v_example = xi_example / (1.0 + xi_example) * gamma_example
    lsa_example = xi_example / (1.0 + xi_example) * math.exp(0.5 * float(exp1(v_example)))
    return {
        "fs_hz": f"{cfg.fs}",
        "nyquist_hz": f"{cfg.nyquist_hz:.0f}",
        "pcm_bits": f"{cfg.pcm_bits}",
        "frame_len": f"{cfg.frame_len}",
        "hop": f"{cfg.hop}",
        "frame_ms": f"{cfg.frame_ms:.0f}",
        "hop_ms": f"{cfg.hop_ms:.0f}",
        "bin_hz": f"{cfg.bin_hz:.2f}",
        "n_bins": f"{cfg.n_bins}",
        "hp_cutoff_hz": f"{cfg.hp_cutoff_hz:.0f}",
        "mains_hz": f"{cfg.mains_hz:.0f}",
        "mains_bin": f"{cfg.mains_hz / cfg.bin_hz:.1f}",
        "dd_alpha": f"{cfg.dd_alpha}",
        "dd_one_minus_alpha": f"{1.0 - cfg.dd_alpha:.2f}",
        "dd_tau_ms": f"{cfg.dd_time_constant_ms:.0f}",
        "ms_n_sub": f"{cfg.ms_n_sub}",
        "ms_sub_frames": f"{cfg.ms_sub_frames}",
        "ms_window_frames": f"{cfg.ms_window_frames}",
        "ms_window_s": f"{cfg.ms_window_s:.3f}",
        "lat_min_samples": f"{cfg.latency_min_samples}",
        "lat_max_samples": f"{cfg.latency_max_samples}",
        "lat_min_ms": f"{1000.0 * cfg.latency_min_samples / cfg.fs:.1f}",
        "lat_max_ms": f"{1000.0 * cfg.latency_max_samples / cfg.fs:.1f}",
        "ms_alpha_s": f"{cfg.ms_alpha_s}",
        "ms_span_min": f"{(cfg.ms_n_sub - 1) * cfg.ms_sub_frames}",
        "ms_span_max": f"{cfg.ms_window_frames - 1}",
        "ms_bias_steady": f"{ms_estimator.bias:.2f}",
        "ms_bias_first": f"{curve[0]:.2f}",
        "ms_bias_mid": f"{curve[cfg.ms_window_frames - 2]:.2f}",
        "mcra_bias": f"{mcra_estimator.bias:.3f}",
        "mcra_alpha_s": f"{cfg.mcra_alpha_s}",
        "mcra_alpha_p": f"{cfg.mcra_alpha_p}",
        "mcra_alpha_d": f"{cfg.mcra_alpha_d}",
        "mcra_delta": f"{cfg.mcra_delta}",
        "mcra_warmup": f"{math.ceil(2.0 / (1.0 - cfg.mcra_alpha_s))}",
        "step_table": step_table,
        "step_down_ms": f"{step_down_ms:.0f}",
        "sqnr_fs_db": f"{quantization_sqnr_db(cfg.pcm_bits):.1f}",
        "sqnr_16_40_db": f"{quantization_sqnr_db(16, 40.0):.1f}",
        "sqnr_8_40_db": f"{quantization_sqnr_db(8, 40.0):.1f}",
        "sqnr_table": _sqnr_table(),
        "frame_ablation_table": _frame_ablation_table(cfg),
        "hp_gain_cut_db": f"{hp_gain(cfg.hp_cutoff_hz):.2f}",
        "hp_gain_mains_db": f"{hp_gain(cfg.mains_hz):.1f}",
        "hp_gain_2mains_db": f"{hp_gain(2 * cfg.mains_hz):.1f}",
        "sine_enbw_bins": f"{sine.enbw_bins:.4f}",
        "sine_null_bins": f"{sine.mainlobe_null_to_null_bins:.1f}",
        "sine_sidelobe_db": f"{sine.peak_sidelobe_db:.1f}",
        "hann_enbw_bins": f"{hann.enbw_bins:.4f}",
        "hann_null_bins": f"{hann.mainlobe_null_to_null_bins:.1f}",
        "hann_sidelobe_db": f"{hann.peak_sidelobe_db:.1f}",
        "sine_null_hz": f"{sine.mainlobe_null_to_null_bins * cfg.bin_hz:.1f}",
        "alias_source_fs": f"{alias.source_fs}",
        "alias_factor": f"{alias.factor}",
        "alias_numtaps": f"{alias.numtaps}",
        "alias_cutoff_hz": f"{alias.cutoff_hz:.0f}",
        "alias_in_hz": f"{alias.in_band_hz:.0f}",
        "alias_out_hz": f"{alias.out_of_band_hz:.0f}",
        "alias_hz": f"{alias.predicted_alias_hz:.0f}",
        "alias_naive_db": f"{alias.naive_alias_db:.1f}",
        "alias_filtered_db": f"{alias.filtered_alias_db:.1f}",
        "lsa_example_xi": f"{xi_example:.0f}",
        "lsa_example_gamma": f"{gamma_example}",
        "lsa_example_gain": f"{lsa_example:.2f}",
    }


def render(template: str, values: dict[str, str]) -> str:
    """Substitute every @@name@@; unknown or unused names raise ValueError."""
    used: set[str] = set()

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise ValueError(f"placeholder @@{name}@@ has no value")
        used.add(name)
        return values[name]

    rendered = PLACEHOLDER.sub(substitute, template)
    unused = sorted(set(values) - used)
    if unused:
        raise ValueError(f"values never used by the template: {', '.join(unused)}")
    return rendered


def generate(root: Path = ROOT) -> str:
    """Render the theory document from its template under `root`."""
    template = (root / TEMPLATE).read_text(encoding="utf-8")
    return render(template, build_values())


def main(argv: list[str], root: Path = ROOT) -> int:
    """`--check` verifies docs/THEORY.md is current; otherwise it is (re)written."""
    rendered = generate(root)
    target = root / OUTPUT
    if "--check" in argv:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != rendered:
            print(f"{OUTPUT} is stale: run `make docs`")
            return 1
        print(f"{OUTPUT} is up to date")
        return 0
    target.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
