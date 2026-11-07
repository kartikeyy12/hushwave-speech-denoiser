"""Build the report: figures and tables from the results, viva notes, then the PDF."""

from __future__ import annotations

import argparse
import sys

from hushwave.reportfigs import make_all as make_figures
from hushwave.reportgen import ClaimError, compile_pdf, write_all, write_viva


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdf", action="store_true", help="only write figures and tables")
    args = parser.parse_args(argv)
    try:
        files = write_all()
        viva = write_viva()
        figures = make_figures()
    except (ClaimError, FileNotFoundError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(files)} files in report/generated, {len(figures)} figures, and {viva}")
    if args.no_pdf:
        return 0
    try:
        pdf = compile_pdf()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
