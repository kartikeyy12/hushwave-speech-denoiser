"""Compare the firmware with the Python engine on a set of signals (make firmware-check).

Builds the firmware sources for this computer, streams the signals through the serial protocol,
and writes results/firmware_check.json, which the report reads.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from hushwave.mcu import build_host, check_firmware

OUTPUT = Path("results") / "firmware_check.json"


def main() -> int:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            report = check_firmware(build_host(Path(tmp)))
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["signals"]:  # type: ignore[attr-defined]
        print(f"{row['signal']:24s} max difference {row['max_abs_difference']:.2e}")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
