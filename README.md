# Kisan Sarthi AI

A sub-800 ms, multilingual (Hindi + code-switch) **voice agent for rural Indian farmers** —
government scheme discovery, KCC/PMFBY walkthroughs, agri-advisory with live tools, and safe
human escalation. Built on the NVIDIA stack (Nemotron, NeMo, TensorRT-LLM, Pipecat) by
**Team IronClad** for the India Agentic AI Open Hackathon.

This repository is the **Gate-0 foundation**: the frozen contracts, a mock for every seam, the
serving lane (vLLM-first, TensorRT-LLM next), the benchmark harness, CI, and an end-to-end driver
— everything needed for four people to build four lanes in parallel without colliding.

---

## The one idea that holds the repo together

The voice path is **async-streaming, not request/response**. ASR, the agent, and TTS are async
generators that yield over time, so TTS can start speaking sentence 1 while the LLM is still
writing sentence 2. Every streamed object carries a `turn_id`, and a barge-in sets an
`asyncio.Event` on the turn so late chunks are dropped.

Four of the five seams are dataclasses in `kisan_sarthi/contracts/models.py`. The **fifth — the
LLM seam — is the OpenAI-compatible HTTP API itself** (`POST /v1/chat/completions`,
`stream=True`). There is no custom `LLMRequest` object to drift out of sync with the real server,
so the mock LLM and the real server (vLLM / TRT-LLM) are interchangeable by a base-URL change.

---

## Quickstart

### 1. Install `uv`

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
**Linux / macOS (the H100 box):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies (no GPU needed for any of Gate 0)
```bash
uv sync            # base + dev deps, creates .venv and uv.lock
```
On the GPU box, add the serving extras: `uv sync --extra gpu` (installs vLLM, ModelOpt).

### 3. Prove Gate 0 is green on your laptop
```bash
uv run pytest -m "not integration" -q     # unit + contract tests
uv run pytest -m integration -q           # integration against mocks
uv run python scripts/gate0_mock.py       # boots mock LLM, runs C2 health-check + C5 bench
```
The last command writes `eval/results/baseline.json`.

### Running on Windows (no `make`)

Windows has no `make` by default. Two options:

- **Use the bundled wrapper** (same target names as the Makefile):
  ```powershell
  ./make.ps1 setup
  ./make.ps1 test
  ./make.ps1 gate0-mock
  ```
- **Or call `uv run` directly** — every Makefile target is just a thin wrapper:
  | Makefile          | Windows / direct                                                    |
  | ----------------- | ------------------------------------------------------------------- |
  | `make setup`      | `uv sync; uv run pre-commit install`                                |
  | `make test`       | `uv run pytest -m "not integration" -q`                             |
  | `make test-integration` | `uv run pytest -m integration -q`                             |
  | `make check-serving` | `uv run python scripts/check_serving.py --url http://localhost:8000` |
  | `make bench`      | `uv run python -m eval.perf.bench --label baseline`                 |
  | `make serve-mock` | `uv run uvicorn app.mocks.mock_llm:app --port 8000`                 |
  | `make e2e`        | `uv run python -m app.main`  (set `PIPELINE`, `KISAN_SERVING_URL`)   |

The Linux H100 box and CI use the **Makefile** directly — it is the canonical interface.

---

## Laptop (mocks) vs. box (real)

Two terminals on the laptop:
```bash
# terminal 1 — the mock LLM (OpenAI-compatible)
make serve-mock                  # or ./make.ps1 serve-mock

# terminal 2 — talk to it
make check-serving               # C2: /health -> 200 + non-empty completion
make bench                       # C5: writes eval/results/baseline.json
PIPELINE=all-mock make e2e       # full turn: mock ASR -> LLM -> mock TTS, with timing
```

On the H100 box, the model becomes real and **nothing else changes** — same endpoint, same
commands. The full step-by-step for the GPU-gated foundational milestones (C2 provision → C4
blueprint loop → C5 real baseline) is in **[BRINGUP.md](BRINGUP.md)**. In short:
```bash
export KISAN_MODEL_ID=<an NVIDIA Nemotron checkpoint>   # model-agnostic; never hard-coded
uv sync --extra gpu
make box-check                   # C2: nvidia-smi + env preflight
make serve                       # launches vLLM on :8000
make check-serving               # same health-check, now hitting the real server
make capture-baseline            # C5: real FP16 baseline -> eval/results/baseline.json
```

---

## Configuration (env)

| Variable             | Meaning                                            | Default                  |
| -------------------- | -------------------------------------------------- | ------------------------ |
| `KISAN_MODEL_ID`     | model name/path sent to serving (model-agnostic)   | `mock-llm`               |
| `KISAN_SERVING_URL`  | server origin (no `/v1`)                           | `http://localhost:8000`  |
| `KISAN_SERVING_BACKEND` | `vllm` (Wk1) or `trtllm` (Wk2)                  | `vllm`                   |
| `PIPELINE`           | `all-mock` or `real` for `app.main`                | `all-mock`               |

Serving knobs (dtype, max-model-len, KV-cache dtype, FP8/NVFP4) live in
`kisan_sarthi/serving/config/serving.yaml`. Precedence: **CLI flag > env > yaml > default.**

---

## Lane ownership

| Lane | Area | Owns (top-level package) |
| ---- | ---- | ------------------------ |
| **1** | Speech I/O (ASR, TTS, transport, barge-in) | `kisan_sarthi/speech/` |
| **2** | Agent & data / RAG / tools / guardrails | `kisan_sarthi/agent/`, `kisan_sarthi/data/` |
| **3** | Serving & optimization | `kisan_sarthi/serving/`, `eval/perf/` |
| **4** | Integration, demo, eval | `app/`, `eval/`, `demo/` |

`kisan_sarthi/contracts/` is **shared and frozen** — changing a shape there is an all-hands, not
a quiet edit.

---

## Repository structure
```
kisan_sarthi/
  contracts/models.py     # the 5 frozen seams (+ JSON (de)serialization)
  speech/                 # Lane 1 (stubs)
  agent/  data/           # Lane 2 (stubs)
  serving/                # Lane 3 — server.py, client.py, quantize.py, trtllm/, metrics.py, config/
app/
  main.py                 # Gate-0 e2e driver (grows into L4.2)
  mocks/                  # mock_llm.py (OpenAI-compatible), mock_asr.py, mock_tts.py
eval/
  perf/bench.py           # the serving benchmark harness (C5)
  results/                # committed metrics JSON (baseline.json, fp16.json, ...)
tests/                    # contract / mock / integration / unit tests
scripts/                  # check_serving.py (C2), gate0_mock.py
Makefile  make.ps1        # task runner (Linux/CI) + Windows wrapper
.github/workflows/ci.yml  # lint + tests on every push (C6)
```

---

## One-time repository setup (you don't have the repo yet)

```bash
cd kisan-sarthi
git init -b main
git add .
git commit -m "Gate 0: contracts, mocks, serving lane, bench, CI, e2e"
```

Create the GitHub repo and push (using the GitHub CLI `gh`, or create it in the web UI first):
```bash
gh repo create team-ironclad/kisan-sarthi --private --source . --remote origin --push
# or, if you made the repo on github.com:
#   git remote add origin https://github.com/team-ironclad/kisan-sarthi.git
#   git push -u origin main
```

**Branch protection (C1)** — on github.com: *Settings → Branches → Add branch ruleset (or
protection rule) for `main`*, and enable:
- Require a pull request before merging (≥ 1 approval)
- Require status checks to pass → select the **CI / lint-and-test** check
- Block force-pushes to `main`

This makes the contracts genuinely frozen: nobody can push a shape change to `main` without a
review and green CI.

---

## Gate-0 status

| # | Item | State | How it's validated |
| - | ---- | ----- | ------------------ |
| **C1** | Monorepo skeleton + `make setup` clean | ✅ now | this repo; `make setup` on a clean machine |
| **C1** | Branch protection on `main` | ⏳ you | one-time GitHub setting (above) — not a code artifact |
| **C2** | Serving health → 200 | ✅ via mock now / 🔜 real on box | `make check-serving` (mock now; real vLLM once the box is up, L3.1) |
| **C3** | Contracts frozen, `test_contracts` green incl. `turn_id` | ✅ now | `uv run pytest -m "not integration"` |
| **C4** | A mock for every seam (ASR / LLM / TTS) | ✅ now | `tests/test_mocks.py` |
| **C5** | Naive baseline latency → `eval/results/baseline.json` | ✅ via mock now / 🔜 FP16 on box | `make bench` / `scripts/gate0_mock.py` (mock numbers now; real FP16-on-H100 lands when the box is up, L3.1) |
| **C6** | CI lint + tests on every push | ✅ now | `.github/workflows/ci.yml` |

**Honest read:** C1 (skeleton), C3, C4, C6 are fully green on a laptop today. C2 and C5 are
*demonstrated end-to-end through the real wire format* against the mock now; the numbers in
`baseline.json` are mock-latency numbers until the GPU box is provisioned, at which point the
identical `make check-serving` / `make bench LABEL=fp16` commands produce the real FP16 baseline
that every later optimization (FP8 in Wk2, NVFP4 in Wk3) is measured against. Branch protection is
the one item that is a GitHub setting rather than code — five clicks once the repo is pushed.
