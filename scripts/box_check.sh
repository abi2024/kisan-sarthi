#!/usr/bin/env bash
# C2 validation: confirm the GPU box is ready before bringing up serving.
# Checks nvidia-smi + required env vars. Run on the H100 box (not the laptop).
set -uo pipefail

fail=0
echo "=== C2 box check ==="

# 1) GPU present
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ok]   nvidia-smi found:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/         /'
else
  echo "[FAIL] nvidia-smi not found — is this the GPU box? Is the NVIDIA driver installed?"
  fail=1
fi

# 2) Required env vars
check_env() {
  if [ -n "${!1:-}" ]; then
    echo "[ok]   \$$1 is set"
  else
    echo "[FAIL] \$$1 is not set — $2"
    fail=1
  fi
}
check_env KISAN_MODEL_ID "export the first model id, e.g. a Nemotron-3-Nano checkpoint or Llama-3.x NIM"
check_env NGC_API_KEY    "create an NGC API key (build.nvidia.com) and export it"

# 3) vLLM available (the GPU extra)
if uv run python -c "import vllm" >/dev/null 2>&1; then
  echo "[ok]   vllm importable (uv sync --extra gpu done)"
else
  echo "[warn] vllm not importable — run: uv sync --extra gpu   (needed for 'make serve')"
fi

echo "==================="
if [ "$fail" -ne 0 ]; then
  echo "C2 NOT READY — fix the [FAIL] lines above, then re-run."
  exit 1
fi
echo "C2 prerequisites OK. Next: 'make serve &' then 'make check-serving'."
