"""Measure the streaming engine: delay, real-time factor, worst block time, memory.

Writes results/engine_performance.json. The numbers depend on this machine and are labelled so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hushwave.enhancer import performance_report

REAL_TIME_FACTOR_TARGET = 0.3


def main(argv: list[str]) -> int:
    report = performance_report()
    out = Path(argv[0]) if argv else Path("results/engine_performance.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"algorithmic delay: {report['algorithmic_delay_samples']} samples "
        f"({report['algorithmic_delay_ms']:.1f} ms); measured: {report['measured_delay_samples']}"
    )
    difference = report["streaming_vs_batch_max_abs_difference"]
    print(f"streaming vs batch max abs difference: {difference:.2e}")
    worst_rtf = 0.0
    for row in report["timing"]:
        worst_rtf = max(worst_rtf, row["real_time_factor"])
        print(
            f"block {row['block']:5d}: RTF {row['real_time_factor']:.3f}, "
            f"worst block {row['worst_block_ms']:.2f} ms of {row['block_budget_ms']:.1f} ms budget"
        )
    print(f"peak heap while streaming 10 s: {report['peak_heap_bytes_10s_stream'] / 1024:.0f} KiB")
    print(f"wrote {out}")
    return 0 if worst_rtf < REAL_TIME_FACTOR_TARGET else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
