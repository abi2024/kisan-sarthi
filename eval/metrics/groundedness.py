"""Groundedness — fraction of an answer's sentences supported by the retrieved sources.

This default is a LEXICAL proxy (content-word overlap), good enough to wire the harness and catch
ungrounded drift on CPU with no model. Swap in an LLM-judge later behind the SAME signature
`groundedness(answer, sources) -> float` (it can call the OpenAI seam like everything else).
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_SENT_SPLIT = re.compile(r"[.!?।]\s*")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text) if s.strip()]


def groundedness(answer: str, sources: list[str], *, threshold: float = 0.5) -> float:
    """Return the fraction of answer sentences whose content-word overlap with the combined
    sources is >= threshold. Empty answer -> 1.0 (nothing ungrounded was said)."""
    src_tokens: set[str] = set()
    for s in sources:
        src_tokens |= _tokens(s)
    sentences = _sentences(answer)
    if not sentences:
        return 1.0
    supported = 0
    for sent in sentences:
        toks = _tokens(sent)
        if not toks:
            supported += 1
            continue
        if len(toks & src_tokens) / len(toks) >= threshold:
            supported += 1
    return supported / len(sentences)


def mean_groundedness(items: list[dict], *, threshold: float = 0.5) -> float:
    """Mean groundedness over [{'answer':..., 'sources':[...]}]."""
    if not items:
        return 0.0
    scores = [
        groundedness(it["answer"], it.get("sources", []), threshold=threshold) for it in items
    ]
    return sum(scores) / len(scores)
