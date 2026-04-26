.PHONY: help install test test-cov run run-sample eda validate clean lint format check-data

PYTHON ?= python
PIP    ?= pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies into the current environment
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev:  ## Install dev tools (pytest, coverage, ruff, black)
	$(PIP) install -r requirements.txt
	$(PIP) install ruff black

check-data:  ## Verify the two Kaggle CSVs are present in data/raw/
	@test -f data/raw/song_lyrics.csv || (echo "❌ data/raw/song_lyrics.csv not found. See README §Setup." && exit 1)
	@test -f data/raw/dataset.csv || (echo "❌ data/raw/dataset.csv not found. See README §Setup." && exit 1)
	@echo "✅ Both raw datasets present."

test:  ## Run the test suite
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

run: check-data  ## Run the full pipeline (~30-90 min)
	$(PYTHON) scripts/run_pipeline.py

run-sample: check-data  ## Run on a 50K-row sample (~2 min smoke test)
	$(PYTHON) scripts/run_pipeline.py --sample 50000

eda:  ## Generate EDA plots and tables in reports/figures/
	$(PYTHON) scripts/eda.py

validate:  ## Validate processed outputs (schema, ranges, leakage)
	$(PYTHON) scripts/validate_output.py

scrape:  ## Fill an underrepresented genre via Genius API (requires .env)
	@if [ -z "$(GENRE)" ]; then echo "Usage: make scrape GENRE=country [N=500]"; exit 1; fi
	$(PYTHON) scripts/scrape_supplemental.py --genre $(GENRE) --n $${N:-500}

lint:  ## Lint with ruff (if installed)
	@command -v ruff >/dev/null 2>&1 && ruff check src/ scripts/ tests/ || echo "ruff not installed; skip"

format:  ## Format with black (if installed)
	@command -v black >/dev/null 2>&1 && black src/ scripts/ tests/ || echo "black not installed; skip"

clean:  ## Remove caches and intermediate artifacts (keeps data/raw/ and data/processed/)
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/interim/* data/cache/* logs/*.log
	@echo "✅ Cleaned caches and intermediates. data/raw/ and data/processed/ untouched."

clean-all: clean  ## Like clean, but also wipes data/processed/ (full re-run from scratch)
	rm -rf data/processed/*
	@echo "✅ Also wiped data/processed/."
