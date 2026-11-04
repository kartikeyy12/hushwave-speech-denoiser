"""Thin wrapper: see hushwave.figures."""

from __future__ import annotations

import sys

from hushwave.figures import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
