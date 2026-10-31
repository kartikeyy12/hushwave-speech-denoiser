"""Download LibriSpeech dev-clean and DEMAND noise, verify checksums, and write the split manifest.

Use --check-urls first to confirm every URL resolves without downloading anything.
"""

from __future__ import annotations

import sys

from hushwave.datasetup import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
