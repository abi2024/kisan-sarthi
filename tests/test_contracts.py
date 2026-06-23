"""C3 — the contract freeze test.

Asserts: every wire contract round-trips to/from JSON; every STREAMED artifact carries a
`turn_id`; TTSChunk audio bytes survive the round-trip; AgentEvent's nested Source
citations survive; and TurnContext is correctly NOT JSON-serializable (it carries an
asyncio.Event). Green here == the seams are frozen and a mock can stand in for the real
thing (Milestone Map §3.1, TESTING.md §2).
"""

import asyncio
import dataclasses

import pytest

from kisan_sarthi.contracts.models import (
    STREAMED_CONTRACTS,
    WIRE_CONTRACTS,
    AgentEvent,
    ASRResult,
    Source,
    TTSChunk,
    from_json,
    new_turn,
    to_json,
)

pytestmark = pytest.mark.contract


def _examples() -> dict[type, object]:
    src = Source(doc="PMFBY Operational Guidelines 2023", locator="p.12 §4.2", snippet="72 hours")
    return {
        Source: src,
        ASRResult: ASRResult(
            turn_id="t1",
            text="aaj mandi bhaav?",
            is_final=True,
            lang="hi",
            t_start=0.0,
            t_end=0.8,
            confidence=0.92,
        ),
        AgentEvent: AgentEvent(
            turn_id="t1",
            kind="text",
            text_delta="PMFBY claim window is 72 hours.",
            citations=[src],
            escalate=False,
            intent="scheme",
        ),
        TTSChunk: TTSChunk(
            turn_id="t1",
            audio_bytes=b"\x00\x01\x02\xff",
            sample_rate=22050,
            is_last=True,
            sentence_idx=3,
        ),
    }


@pytest.mark.parametrize("cls", WIRE_CONTRACTS, ids=lambda c: c.__name__)
def test_wire_contract_round_trips_through_json(cls):
    original = _examples()[cls]
    restored = from_json(cls, to_json(original))
    assert restored == original, f"{cls.__name__} did not survive JSON round-trip"


@pytest.mark.parametrize("cls", STREAMED_CONTRACTS, ids=lambda c: c.__name__)
def test_every_streamed_contract_carries_turn_id(cls):
    field_names = {f.name for f in dataclasses.fields(cls)}
    assert "turn_id" in field_names, f"{cls.__name__} is missing turn_id"
    instance = _examples()[cls]
    assert instance.turn_id, f"{cls.__name__} instance has empty turn_id"


def test_ttschunk_audio_bytes_preserved():
    chunk = _examples()[TTSChunk]
    restored = from_json(TTSChunk, to_json(chunk))
    assert isinstance(restored.audio_bytes, bytes)
    assert restored.audio_bytes == chunk.audio_bytes


def test_agentevent_nested_citations_preserved():
    ev = _examples()[AgentEvent]
    restored = from_json(AgentEvent, to_json(ev))
    assert restored.citations == ev.citations
    assert all(isinstance(c, Source) for c in restored.citations)


def test_turncontext_is_not_json_serializable():
    ctx = new_turn(session_id="s1", lang="hi")
    assert ctx.turn_id and isinstance(ctx.cancel, asyncio.Event)
    with pytest.raises(TypeError):
        to_json(ctx)
