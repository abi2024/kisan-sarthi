"""Level-1 unit tests for the benchmark statistics (pure functions, no server)."""

import math
import shutil

import pytest

from eval.perf.bench import RequestResult, percentile, summarize


def test_percentile_basic():
    data = [10, 20, 30, 40, 50]
    assert percentile(data, 50) == 30
    assert percentile(data, 0) == 10
    assert percentile(data, 100) == 50


def test_percentile_interpolates():
    # p95 of 1..100 (linear interpolation) is 95.05
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05, abs=1e-9)


def test_percentile_empty_is_nan():
    assert math.isnan(percentile([], 50))


def test_percentile_single_value():
    assert percentile([42.0], 95) == 42.0


def test_summarize_shapes_and_aggregate():
    results = [
        RequestResult(ttft_ms=50, total_ms=200, output_tokens=20, ok=True),
        RequestResult(ttft_ms=60, total_ms=300, output_tokens=30, ok=True),
        RequestResult(ttft_ms=0, total_ms=10, output_tokens=0, ok=False, error="boom"),
    ]
    s = summarize(
        results,
        label="baseline",
        model="mock-llm",
        endpoint="http://x",
        concurrency=2,
        wall_time_s=0.5,
        gpu=None,
    )
    assert s["n_requests"] == 3 and s["n_ok"] == 2 and s["n_error"] == 1
    assert set(s["ttft_ms"]) == {"p50", "p95", "mean"}
    # aggregate tokens/sec = 50 tokens / 0.5 s = 100
    assert s["tokens_per_sec"]["aggregate"] == pytest.approx(100.0)
    assert s["gpu_util_pct"] is None and s["gpu"] is None


def test_summarize_gpu_block_headline_is_peak():
    gpu = {"util_mean_pct": 40.0, "util_peak_pct": 92.0, "mem_used_peak_mb": 18000.0, "samples": 12}
    s = summarize(
        [RequestResult(ttft_ms=10, total_ms=20, output_tokens=5, ok=True)],
        label="fp16",
        model="m",
        endpoint="e",
        concurrency=1,
        wall_time_s=0.1,
        gpu=gpu,
    )
    assert s["gpu_util_pct"] == 92.0  # headline = peak under load
    assert s["gpu"]["util_mean_pct"] == 40.0 and s["gpu"]["samples"] == 12


def test_gpu_sampler_degrades_to_none_without_gpu():
    from eval.perf.bench import GpuSampler

    with GpuSampler() as g:
        pass
    # On a box with no nvidia-smi (CI/laptop), summary is None and nothing raised.
    if shutil.which("nvidia-smi") is None:
        assert g.summary() is None
