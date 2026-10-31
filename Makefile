SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PYTHON ?= $(shell bash scripts/find_python.sh)

.PHONY: setup data test lint typecheck experiments figures report demo-live demo-replay verify all clean

setup:
	@test -n "$(PYTHON)" || { echo "No suitable Python found. Install one or pass PYTHON=/path/to/python"; exit 1; }
	"$(PYTHON)" -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m pip freeze --exclude-editable > requirements.lock
	@$(PY) --version
	@echo "setup complete; requirements.lock written from the real pip freeze"

data:
	@echo "make data is added in milestone M3"; exit 1

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src scripts tests
	$(PY) -m ruff format --check src scripts tests

typecheck:
	$(PY) -m mypy

experiments:
	@echo "make experiments is added in milestone M8"; exit 1

figures:
	@echo "make figures is added in milestone M8"; exit 1

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
