"""Level-1 unit tests for the eval metrics (pure functions) and the harness wiring."""

import math

import pytest

from eval.metrics.groundedness import groundedness, mean_groundedness
from eval.metrics.latency import percentile, summarize_latencies
from eval.metrics.wer import corpus_wer, wer


def test_wer_perfect_and_empty():
    assert wer("a b c", "a b c") == 0.0
    assert wer("", "") == 0.0
    assert wer("", "spurious") == 1.0


def test_wer_one_substitution():
    assert wer("the cat sat down", "the dog sat down") == pytest.approx(0.25)


def test_corpus_wer_aggregates_by_total_words():
    pairs = [("a b", "a b"), ("c d e f", "c x e f")]
    assert corpus_wer(pairs) == pytest.approx(1 / 6)


def test_groundedness_supported_vs_unsupported():
    supported = groundedness(
        "Claims report within 72 hours.", ["report crop loss within 72 hours via bank"]
    )
    assert supported == 1.0
    unsupported = groundedness(
        "You will win the lottery tomorrow.", ["crop loss reporting window is 72 hours"]
    )
    assert unsupported == 0.0


def test_groundedness_empty_answer_is_grounded():
    assert groundedness("", ["anything"]) == 1.0


def test_mean_groundedness_range():
    items = [
        {"answer": "report within 72 hours", "sources": ["report within 72 hours"]},
        {"answer": "totally made up nonsense", "sources": ["unrelated text"]},
    ]
    score = mean_groundedness(items)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.5)


def test_latency_percentile_and_summary():
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05, abs=1e-9)
    assert math.isnan(percentile([], 50))
    s = summarize_latencies([100, 200, 300])
    assert s["p50"] == 200 and s["n"] == 3


def test_harness_runs_against_fixtures():
    import eval.harness as h

    report = h.Harness().run_all()
    suites = report["suites"]
    assert suites["ASR WER"]["status"] == "ok"
    assert suites["RAG groundedness"]["status"] == "ok"
    assert suites["Intent routing"]["status"] == "ok"
    assert suites["Intent routing"]["detail"]["accuracy"] == pytest.approx(0.8)
    assert (h.RESULTS_DIR / "report.md").exists()
