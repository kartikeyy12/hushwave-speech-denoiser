"""Write firmware/hushwave_esp32/hushwave_params.h from the tuning file (make firmware-params)."""

from __future__ import annotations

import sys

from hushwave.mcu import write_params


def main() -> int:
    try:
        path = write_params()
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
