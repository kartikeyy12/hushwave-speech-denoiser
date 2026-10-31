"""Regenerate docs/THEORY.md from docs/THEORY.template.md (use --check to only verify)."""

from __future__ import annotations

import sys

from hushwave.docfill import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
