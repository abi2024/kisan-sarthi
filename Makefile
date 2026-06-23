# Kisan Sarthi AI — the single entrypoint for build / test / serve / demo.
# Canonical on the Linux H100 box and in CI. On Windows, use make.ps1 (same targets)
# or the `uv run ...` equivalents documented in the README.

SHELL := /bin/bash

# Server origin used by check-serving / bench / e2e (mock or real, same endpoint).
KISAN_SERVING_URL ?= http://localhost:8000
PORT ?= 8000
PIPELINE ?= all-mock
LABEL ?= baseline
CONCURRENCY ?= 1

.DEFAULT_GOAL := help
.PHONY: help setup lint fmt test test-integration check-serving bench \
        serve-mock serve e2e demo gate0-mock box-check capture-baseline clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Clean-machine bootstrap (uv venv + deps + pre-commit hooks)
	uv sync
	uv run pre-commit install
	@echo "setup complete. Activate with: source .venv/bin/activate (Linux/Mac)"

lint: ## ruff + black --check (no changes)
	uv run ruff check .
	uv run black --check .

fmt: ## Auto-format (black) and auto-fix (ruff)
	uv run ruff check --fix .
	uv run black .

test: ## Level 1+2: unit + contract tests (no GPU)
	uv run pytest -m "not integration" -q

test-integration: ## Level 3: integration against mocks (no GPU, CI)
	uv run pytest -m integration -q

check-serving: ## C2: health check -> 200 + non-empty completion (probes $KISAN_SERVING_URL)
	uv run python scripts/check_serving.py --url $(KISAN_SERVING_URL)

bench: ## C5: run bench.py against the serving endpoint -> eval/results/<label>.json
	uv run python -m eval.perf.bench --label $(LABEL) --concurrency $(CONCURRENCY) \
	  --endpoint $(KISAN_SERVING_URL)

serve-mock: ## Launch the OpenAI-compatible MOCK LLM (laptop, no GPU)
	uv run uvicorn app.mocks.mock_llm:app --host 127.0.0.1 --port $(PORT) --log-level warning

serve: ## Launch the REAL serving endpoint (GPU box; needs KISAN_MODEL_ID + uv sync --extra gpu)
	uv run python -m kisan_sarthi.serving.server

e2e: ## Full pipeline shape; backend via PIPELINE=all-mock|real
	PIPELINE=$(PIPELINE) KISAN_SERVING_URL=$(KISAN_SERVING_URL) uv run python -m app.main

demo: ## The presentable run (wraps e2e with the UI — Lane 4, Wk4)
	@echo "demo UI lands in L4.4 (Wk4). For now: make e2e PIPELINE=all-mock"

gate0-mock: ## Boot mock LLM, run check-serving + a small bench, tear down (laptop Gate-0 proof)
	uv run python scripts/gate0_mock.py

box-check: ## C2 (GPU box only): verify nvidia-smi + NGC/model env before serving — see BRINGUP.md
	bash scripts/box_check.sh

capture-baseline: ## C5 (GPU box only): capture real FP16 baseline against a running 'make serve'
	bash scripts/capture_baseline.sh

clean: ## Remove caches and the venv
	rm -rf .venv .pytest_cache .ruff_cache **/__pycache__ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
