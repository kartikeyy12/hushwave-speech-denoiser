"""Thin wrapper: see hushwave.resultsdoc."""

from __future__ import annotations

import sys

from hushwave.resultsdoc import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
