"""Build the replay clips in examples/ from your own recordings (make examples)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hushwave.examples import build_examples


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=Path("data/recordings"))
    parser.add_argument("--out", type=Path, default=Path("examples"))
    args = parser.parse_args(argv)
    try:
        manifest = build_examples(args.recordings, args.out)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for clip in manifest["clips"]:
        print(f"{clip['id']}: {clip['duration_s']:.1f} s")
    print(f"wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
