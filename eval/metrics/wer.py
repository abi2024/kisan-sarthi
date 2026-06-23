"""Word Error Rate (and code-switch WER) — pure, no I/O. Used by the eval harness (L4.3)."""

from __future__ import annotations

from collections.abc import Iterable


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance (substitutions + insertions + deletions)."""
    m, n = len(ref), len(hyp)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def wer(reference: str, hypothesis: str) -> float:
    """WER for one utterance = edit_distance(words) / len(reference words)."""
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def corpus_wer(pairs: Iterable[tuple[str, str]]) -> float:
    """Aggregate WER the correct way: total edits / total reference words (not mean of per-utt)."""
    total_edits = 0
    total_words = 0
    for reference, hypothesis in pairs:
        ref = reference.split()
        total_edits += _edit_distance(ref, hypothesis.split())
        total_words += len(ref)
    if total_words == 0:
        return 0.0
    return total_edits / total_words
