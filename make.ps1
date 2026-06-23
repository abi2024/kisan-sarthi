# make.ps1 — Windows wrapper for the Makefile targets (Windows has no `make` by default).
# Usage:  ./make.ps1 <target>   e.g.  ./make.ps1 test
param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
if (-not $env:KISAN_SERVING_URL) { $env:KISAN_SERVING_URL = "http://localhost:8000" }
$port = if ($env:KISAN_SERVING_PORT) { $env:KISAN_SERVING_PORT } else { "8000" }
$pipeline = if ($env:PIPELINE) { $env:PIPELINE } else { "all-mock" }
$label = if ($env:LABEL) { $env:LABEL } else { "baseline" }
$conc = if ($env:CONCURRENCY) { $env:CONCURRENCY } else { "1" }

switch ($Target) {
    "setup"            { uv sync; uv run pre-commit install }
    "lint"            { uv run ruff check .; uv run black --check . }
    "fmt"             { uv run ruff check --fix .; uv run black . }
    "test"            { uv run pytest -m "not integration" -q }
    "test-integration" { uv run pytest -m integration -q }
    "check-serving"   { uv run python scripts/check_serving.py --url $env:KISAN_SERVING_URL }
    "bench"           { uv run python -m eval.perf.bench --label $label --concurrency $conc --endpoint $env:KISAN_SERVING_URL }
    "serve-mock"      { uv run uvicorn app.mocks.mock_llm:app --host 127.0.0.1 --port $port --log-level warning }
    "serve"           { uv run python -m kisan_sarthi.serving.server }
    "e2e"             { $env:PIPELINE = $pipeline; uv run python -m app.main }
    "gate0-mock"      { uv run python scripts/gate0_mock.py }
    "clean"           { Remove-Item -Recurse -Force .venv, .pytest_cache, .ruff_cache -ErrorAction SilentlyContinue }
    default {
        Write-Host "Targets: setup, lint, fmt, test, test-integration, check-serving, bench, serve-mock, serve, e2e, gate0-mock, clean"
    }
}
