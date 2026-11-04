SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PYTHON ?= $(shell bash scripts/find_python.sh)

.PHONY: setup data data-check test lint typecheck docs d1-figure perf experiments experiments-quick figures results-doc report demo-live demo-replay verify all clean

setup:
	@test -n "$(PYTHON)" || { echo "No suitable Python found. Install one or pass PYTHON=/path/to/python"; exit 1; }
	"$(PYTHON)" -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m pip freeze --exclude-editable > requirements.lock
	@$(PY) --version
	@echo "setup complete; requirements.lock written from the real pip freeze"

data:
	$(PY) scripts/download_data.py

data-check:
	$(PY) scripts/download_data.py --check-urls

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src scripts tests
	$(PY) -m ruff format --check src scripts tests

typecheck:
	$(PY) -m mypy

docs:
	$(PY) scripts/fill_docs.py

d1-figure:
	$(PY) scripts/make_aliasing_figure.py

perf:
	$(PY) scripts/measure_engine.py

# M8. Recordings are read from data/recordings (raw WAVs of any rate or channel count are fine).
# experiments = tune on DEV, evaluate once on TEST, score the real recordings (see docs/RESULTS.md).
# Pass WORKERS=n to change the number of processes, e.g. `make experiments WORKERS=8`.
WORKERS ?= $(shell $(PY) -c "import os; print(max(1, (os.cpu_count() or 2) - 2))")

experiments:
	$(PY) scripts/run_experiments.py --stage all --profile full --workers $(WORKERS)

experiments-quick:
	$(PY) scripts/run_experiments.py --stage all --profile quick --workers $(WORKERS) --results-dir results/quick

figures:
	$(PY) scripts/make_figures.py

results-doc:
	$(PY) scripts/make_results_doc.py

report:
	@echo "make report is added in milestone M10"; exit 1

demo-live:
	@echo "make demo-live is added in milestone M9"; exit 1

demo-replay:
	@echo "make demo-replay is added in milestone M9"; exit 1

verify:
	$(PY) scripts/verify.py

all: verify

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis .coverage htmlcov build dist src/*.egg-info
