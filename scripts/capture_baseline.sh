#!/usr/bin/env bash
# C5: capture the naive BF16 serving baseline against a RUNNING real endpoint.
# Assumes 'make serve' is up on $KISAN_SERVING_URL. Writes baseline.json (canonical,
# concurrency=1) + l3_1_bf16.json, plus a 1/4/8 concurrency sweep for L3.3 to beat.
set -euo pipefail

URL="${KISAN_SERVING_URL:-http://localhost:8000}"
N="${BENCH_N:-50}"
MAXTOK="${BENCH_MAX_TOKENS:-128}"
mkdir -p eval/results

echo "=== C5 baseline capture: endpoint=$URL n=$N ==="

# Pre-flight: endpoint must be healthy first.
uv run python scripts/check_serving.py --url "$URL"

# Canonical naive baseline: single stream.
uv run python -m eval.perf.bench --label bf16 --concurrency 1 --n "$N" \
  --max-tokens "$MAXTOK" --endpoint "$URL" --out eval/results/baseline.json
cp eval/results/baseline.json eval/results/l3_1_bf16.json

# Concurrency sweep (L3.3 batching reference curve).
for c in 4 8; do
  uv run python -m eval.perf.bench --label "bf16_c${c}" --concurrency "$c" --n "$N" \
    --max-tokens "$MAXTOK" --endpoint "$URL" --out "eval/results/bf16_c${c}.json"
done

echo ""
echo "=== wrote ==="
ls -1 eval/results/baseline.json eval/results/l3_1_bf16.json eval/results/bf16_c*.json
echo ""
echo "C5 done. Commit baseline.json — every optimization is a delta vs this."
