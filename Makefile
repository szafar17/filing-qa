# Filing Q&A: common commands. Run `make help` for the list.
PY ?= .venv/bin/python
CONFIG ?= config/default.yaml
OFFLINE ?= config/offline.yaml

.PHONY: help install up down migrate download index embed ingest stats serve \
        draft-golden label eval thresholds test test-unit lint format \
        offline-ingest eval-offline ci

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## create .venv and install with dev + local-model extras
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,local]"

up: ## start Postgres + pgvector (docker compose)
	docker compose up -d --wait

down: ## stop Postgres (data volume is kept)
	docker compose down

migrate: ## create/upgrade the schema
	$(PY) -m filingqa.store --config $(CONFIG) migrate

download: ## download 10-K/10-Q filings from EDGAR (needs SEC_USER_AGENT)
	$(PY) -m filingqa.ingest --config $(CONFIG) download

index: ## parse, chunk and load downloaded filings; refresh BM25 stats
	$(PY) -m filingqa.ingest --config $(CONFIG) index

embed: ## embed chunks that have no vector yet; build the HNSW index
	$(PY) -m filingqa.ingest --config $(CONFIG) embed

ingest: migrate download index embed ## full pipeline: migrate + download + index + embed

stats: ## corpus statistics
	$(PY) -m filingqa.ingest --config $(CONFIG) stats

serve: ## run the API on :8000
	FILINGQA_CONFIG=$(CONFIG) $(PY) -m uvicorn filingqa.api.app:app --port 8000

draft-golden: ## LLM-draft golden-set candidates from the index (unverified)
	$(PY) -m filingqa.eval.draft --config $(CONFIG)

label: ## review drafts in the terminal; accepted items become human-verified
	$(PY) -m filingqa.eval.label --config $(CONFIG) --show-retrieval

eval: ## official eval (needs >= 100 human-verified items); gated by eval/thresholds.json
	$(PY) -m filingqa.eval --config $(CONFIG)

thresholds: ## re-baseline eval/thresholds.json from results/latest.json
	$(PY) -m filingqa.eval.gate results/latest.json eval/thresholds.json --write

test: ## all tests (integration tests need FILINGQA_TEST_DATABASE_URL)
	$(PY) -m pytest -q

test-unit: ## tests that need no database
	$(PY) -m pytest -q -m "not integration"

lint: ## ruff lint + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

offline-ingest: ## load the synthetic mini corpus with offline components
	$(PY) -m filingqa.store --config $(OFFLINE) migrate
	$(PY) -m filingqa.ingest --config $(OFFLINE) local tests/fixtures/corpus
	$(PY) -m filingqa.ingest --config $(OFFLINE) embed

eval-offline: offline-ingest ## smoke eval on synthetic data (what CI runs), gated by thresholds.offline.json
	$(PY) -m filingqa.eval --config $(OFFLINE) --smoke

ci: lint test eval-offline ## everything CI runs
