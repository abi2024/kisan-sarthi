"""Latency helpers for the eval harness — percentiles over end-to-end turn latencies, plus a
loader for the serving bench JSON. Pure (small local percentile to avoid heavy imports)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(s[int(k)])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def summarize_latencies(samples_ms: Sequence[float]) -> dict:
    """p50/p95/mean over a list of end-to-end turn latencies (ms)."""
    return {
        "p50": round(percentile(samples_ms, 50), 2),
        "p95": round(percentile(samples_ms, 95), 2),
        "mean": round(sum(samples_ms) / len(samples_ms), 2) if samples_ms else float("nan"),
        "n": len(samples_ms),
    }


def load_bench_json(path: str | Path) -> dict | None:
    """Load a serving bench result (eval/results/*.json); None if missing/unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
