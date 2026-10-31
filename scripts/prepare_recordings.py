"""Check and convert your own recordings to 16 kHz mono 16-bit WAV in data/recordings/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hushwave.recordings import prepare_recording


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/recordings"))
    parser.add_argument("--min-duration", type=float, default=30.0)
    args = parser.parse_args(argv)
    problems = 0
    for source in args.files:
        destination = args.out / (source.stem + ".wav")
        report = prepare_recording(source, destination, args.min_duration)
        print(
            f"{source.name}: {report.duration_s:.1f} s, rms {report.rms_dbfs:.1f} dBFS, "
            f"peak {report.peak_dbfs:.1f} dBFS, "
            f"quiet-frame level {report.quiet_frame_dbfs:.1f} dBFS"
        )
        for warning in report.warnings:
            print(f"  WARNING: {warning}")
            problems += 1
        print(f"  saved {destination}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
