"""eval/perf/bench.py — the serving benchmark harness (C5, and the optimization story).

Hits the OpenAI-compatible endpoint with a fixed prompt set at fixed concurrency and
measures time-to-first-token, tokens/sec, p50/p95 latency and GPU utilization, then
writes one JSON to eval/results/. Run the SAME harness against the mock, FP16, FP8 and
NVFP4 — the before/after speedup is then a real, reproducible number (one JSON per
precision; `--label` names it).

    python -m eval.perf.bench --label baseline --concurrency 1
    python -m eval.perf.bench --label fp16 --concurrency 4 --out eval/results/l3_1_fp16.json

Because the seam is the OpenAI HTTP API, this needs nothing but an HTTP client and a
prompt file — no speech pipeline, no GPU required to run (GPU util is recorded when a
GPU is present, null otherwise). The measurement is async; the statistics are pure
functions (`percentile`, `summarize`) so they're unit-tested without a server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kisan_sarthi.serving.client import DEFAULT_MODEL, make_client

_DEFAULT_PROMPTS = Path(__file__).parent / "bench_prompts.txt"
_DEFAULT_OUT = Path("eval/results/baseline.json")


@dataclass
class RequestResult:
    ttft_ms: float
    total_ms: float
    output_tokens: int
    ok: bool
    error: str | None = None


# --------------------------------------------------------------------------- #
# Pure statistics (no I/O — unit-tested in tests/test_bench_math.py)
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0, 100]). NaN for empty input."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(s[int(k)])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "mean": round(sum(values) / len(values), 2) if values else float("nan"),
    }


def summarize(
    results: Sequence[RequestResult],
    *,
    label: str,
    model: str,
    endpoint: str,
    concurrency: int,
    wall_time_s: float,
    gpu: dict | None,
) -> dict:
    """Turn raw per-request results into the committed metrics JSON. Pure."""
    ok = [r for r in results if r.ok]
    ttfts = [r.ttft_ms for r in ok]
    totals = [r.total_ms for r in ok]
    per_req_tps = [
        r.output_tokens / (r.total_ms / 1000.0) for r in ok if r.total_ms > 0 and r.output_tokens
    ]
    total_out_tokens = sum(r.output_tokens for r in ok)
    return {
        "label": label,
        "model": model,
        "endpoint": endpoint,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "concurrency": concurrency,
        "n_requests": len(results),
        "n_ok": len(ok),
        "n_error": len(results) - len(ok),
        "wall_time_s": round(wall_time_s, 3),
        "ttft_ms": _stats(ttfts),
        "total_latency_ms": _stats(totals),
        "tokens_per_sec": {
            "mean_per_request": (
                round(sum(per_req_tps) / len(per_req_tps), 2) if per_req_tps else float("nan")
            ),
            "aggregate": (
                round(total_out_tokens / wall_time_s, 2) if wall_time_s > 0 else float("nan")
            ),
        },
        # Headline = PEAK util under load (mean is diluted by ramp-up/teardown).
        "gpu_util_pct": gpu["util_peak_pct"] if gpu else None,
        "gpu": gpu,
    }


# --------------------------------------------------------------------------- #
# Measurement (async I/O against the endpoint)
# --------------------------------------------------------------------------- #


async def run_one(client, model: str, prompt: str, max_tokens: int) -> RequestResult:
    messages = [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    counted = 0
    usage_completion: int | None = None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            extra_body={"reasoning": False},
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                usage_completion = getattr(usage, "completion_tokens", None)
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                counted += 1
        total_ms = (time.perf_counter() - t0) * 1000.0
        # Prefer exact server-reported completion tokens; fall back to counted chunks.
        out_tokens = usage_completion if usage_completion is not None else counted
        return RequestResult(ttft_ms or total_ms, total_ms, out_tokens, ok=True)
    except Exception as e:  # noqa: BLE001 — record, don't crash the whole run
        total_ms = (time.perf_counter() - t0) * 1000.0
        return RequestResult(total_ms, total_ms, 0, ok=False, error=f"{type(e).__name__}: {e}")


async def run_bench(
    prompts: Sequence[str],
    *,
    concurrency: int,
    model: str,
    max_tokens: int,
    origin: str,
    http_client=None,
) -> tuple[list[RequestResult], float, dict | None]:
    client = make_client(origin, http_client=http_client)
    sem = asyncio.Semaphore(concurrency)

    async def worker(p: str) -> RequestResult:
        async with sem:
            return await run_one(client, model, p, max_tokens)

    with GpuSampler() as gpu:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(worker(p) for p in prompts))
        wall = time.perf_counter() - t0
    await client.close()
    return list(results), wall, gpu.summary()


class GpuSampler:
    """Background GPU sampler.

    Polls `nvidia-smi` on a timer in a daemon thread WHILE the benchmark runs, so we
    capture utilization under load and report mean + peak — not a single post-run
    snapshot, which lands after generation has stopped and under-reports. Degrades to a
    no-op (summary() -> None) when no GPU / nvidia-smi is present (laptop, CI).

    Util is averaged across GPUs per sample; memory is summed across GPUs per sample.
    """

    _QUERY = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._util: list[float] = []
        self._mem: list[float] = []
        self._available = shutil.which("nvidia-smi") is not None

    def _poll_once(self) -> None:
        out = subprocess.check_output(self._QUERY, text=True, timeout=5)
        utils, mems = [], []
        for line in out.strip().splitlines():
            u, m = (p.strip() for p in line.split(","))
            utils.append(float(u))
            mems.append(float(m))
        if utils:
            self._util.append(sum(utils) / len(utils))
        if mems:
            self._mem.append(sum(mems))  # total VRAM in use across GPUs

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuSampler:
        if not self._available:
            return self
        try:
            self._poll_once()  # prime one sample; if this fails, mark unavailable
        except Exception:
            self._available = False
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict | None:
        if not self._util:
            return None
        return {
            "util_mean_pct": round(sum(self._util) / len(self._util), 1),
            "util_peak_pct": round(max(self._util), 1),
            "mem_used_peak_mb": round(max(self._mem), 1) if self._mem else None,
            "samples": len(self._util),
        }


def load_prompts(path: Path, n: int | None) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"No prompts found in {path}")
    if n is None:
        return lines
    # Repeat/cycle the prompt set to reach exactly n requests.
    return [lines[i % len(lines)] for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Serving benchmark harness (C5)")
    parser.add_argument(
        "--label", default="baseline", help="precision/run label (also default filename)"
    )
    parser.add_argument("--dtype", default=None, help="alias for --label (kept for L3.1 naming)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--n", type=int, default=None, help="total requests (default: one per prompt)"
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--model", default=None, help="model name (else KISAN_MODEL_ID / mock-llm)")
    parser.add_argument("--endpoint", default=None, help="server origin (else KISAN_SERVING_URL)")
    parser.add_argument("--prompts", default=str(_DEFAULT_PROMPTS))
    parser.add_argument(
        "--out", default=None, help="output JSON path (default eval/results/<label>.json)"
    )
    args = parser.parse_args()

    label = args.dtype or args.label
    model = args.model or DEFAULT_MODEL
    origin = args.endpoint or os.getenv("KISAN_SERVING_URL", "http://localhost:8000")
    prompts = load_prompts(Path(args.prompts), args.n)
    out_path = (
        Path(args.out)
        if args.out
        else (_DEFAULT_OUT if label == "baseline" else Path(f"eval/results/{label}.json"))
    )

    print(
        f"[bench] label={label} model={model} endpoint={origin} "
        f"concurrency={args.concurrency} requests={len(prompts)}",
        flush=True,
    )

    results, wall, gpu = asyncio.run(
        run_bench(
            prompts,
            concurrency=args.concurrency,
            model=model,
            max_tokens=args.max_tokens,
            origin=origin,
        )
    )
    summary = summarize(
        results,
        label=label,
        model=model,
        endpoint=origin,
        concurrency=args.concurrency,
        wall_time_s=wall,
        gpu=gpu,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\n[bench] wrote {out_path}")
    if summary["n_error"]:
        print(
            f"[bench] WARNING: {summary['n_error']} request(s) failed "
            f"(first error: {next((r.error for r in results if not r.ok), None)})"
        )


if __name__ == "__main__":
    main()
