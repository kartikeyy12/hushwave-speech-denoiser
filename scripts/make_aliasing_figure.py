"""Produce the D1 aliasing figure: decimation with and without an anti-alias filter."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hushwave.theory import aliasing_experiment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "figures" / "d1_aliasing.png"


def main() -> None:
    result = aliasing_experiment()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = (
        (axes[0], result.naive_spectrum_db, "Decimation without anti-alias filter"),
        (axes[1], result.filtered_spectrum_db, "Decimation with FIR anti-alias filter"),
    )
    for axis, spectrum, title in panels:
        axis.plot(result.freqs_hz / 1000.0, spectrum, linewidth=0.9)
        axis.axvline(result.in_band_hz / 1000.0, color="green", linestyle=":", label="in-band tone")
        axis.axvline(
            result.predicted_alias_hz / 1000.0, color="red", linestyle="--", label="predicted alias"
        )
        axis.set_title(title)
        axis.set_xlabel("Frequency (kHz)")
        axis.set_ylim(-120, 10)
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right")
    axes[0].set_ylabel("Level relative to in-band tone (dB)")
    fig.suptitle(
        f"{result.source_fs} Hz to {result.target_fs} Hz: tone at {result.out_of_band_hz:.0f} Hz "
        f"aliases to {result.predicted_alias_hz:.0f} Hz "
        f"(alias {result.naive_alias_db:.1f} dB vs {result.filtered_alias_db:.1f} dB)"
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
