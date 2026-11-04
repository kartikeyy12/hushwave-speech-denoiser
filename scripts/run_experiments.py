"""M8: tune on DEV, evaluate once on TEST, score the real recordings -> results/results.json.

Examples:
    python scripts/run_experiments.py --stage all                  # the reported run
    python scripts/run_experiments.py --stage all --profile quick  # a rehearsal
"""

from __future__ import annotations

import sys

from hushwave.pipeline import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
