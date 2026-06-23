"""Kisan Sarthi AI — the five interface contracts (the seams between lanes).

SHARED AND FROZEN. Every lane imports from here; nobody re-defines these locally
(`grep` for local re-definitions must come back empty — see Milestone Map §3.1).

Two principles run through all of them:
  1. The voice path is ASYNC-STREAMING, not request/response. Producers (ASR, the
     agent, TTS) yield over time, so TTS can start on sentence 1 while the LLM is
     still generating sentence 2. This is what makes the <800 ms budget possible.
  2. `turn_id` EVERYWHERE + first-class cancellation. Every streamed artifact carries
     the turn it belongs to, so a barge-in can cancel turn N and the system drops its
     late chunks. `TurnContext` carries an `asyncio.Event` the barge-in controller sets.

The LLM seam (Lane 2 -> Lane 3) is deliberately NOT a dataclass here: it IS the
OpenAI-compatible HTTP API (`POST /v1/chat/completions`, `stream=True`,
`extra_body={"reasoning": ..., "max_tokens": ...}`, cancel via client abort). There is
no parallel `LLMRequest` object to drift out of sync with the real server.

Changing any shape in this file is a mini all-hands, not a quiet edit (TESTING.md §7).
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

# --------------------------------------------------------------------------- #
# The contracts (shapes are frozen — field names, types, defaults, order)
# --------------------------------------------------------------------------- #


@dataclass
class Source:
    doc: str  # e.g. 'PMFBY Operational Guidelines 2023'
    locator: str  # page / section / url
    snippet: str


@dataclass
class TurnContext:
    # Producer -> Consumer: Lane 4 -> all. In-process control object (NOT a wire
    # artifact — it carries an asyncio.Event and is never JSON-serialized).
    turn_id: str
    session_id: str
    lang: str
    cancel: asyncio.Event  # barge-in sets this; agent + LLM client check it and abort
    t_started: float


@dataclass
class ASRResult:
    # Producer -> Consumer: Lane 1 -> Lane 2. Streamed (interim + final).
    turn_id: str
    text: str
    is_final: bool
    lang: str
    t_start: float
    t_end: float
    confidence: float


@dataclass
class AgentEvent:
    # Producer -> Consumer: Lane 2 -> Lane 1/4. Streamed, not a single blob: a turn is
    # a stream of events (partial text, tool calls, final marker, escalation), so Lane 1
    # can speak as the answer forms. Replaces the old single `AgentResponse`.
    turn_id: str
    kind: str  # 'text' | 'tool' | 'final' | 'escalate'
    text_delta: str = ""
    citations: list[Source] = field(default_factory=list)
    escalate: bool = False
    intent: str = "unknown"


@dataclass
class TTSChunk:
    # Producer -> Consumer: Lane 2 -> Lane 1. Streamed per sentence so playback starts
    # before the full reply is generated.
    turn_id: str
    audio_bytes: bytes
    sample_rate: int = 22050
    is_last: bool = False
    sentence_idx: int = 0


# The streamed wire artifacts (each MUST carry turn_id). Used by test_contracts.py.
STREAMED_CONTRACTS: tuple[type, ...] = (ASRResult, AgentEvent, TTSChunk)
# Everything that crosses a lane boundary as JSON (TurnContext is excluded on purpose).
WIRE_CONTRACTS: tuple[type, ...] = (Source, ASRResult, AgentEvent, TTSChunk)


# --------------------------------------------------------------------------- #
# JSON (de)serialization for the wire contracts.
#
# `bytes` (TTSChunk.audio_bytes) is base64-encoded under a sentinel key so audio
# survives a JSON round-trip. TurnContext is intentionally not serializable.
# --------------------------------------------------------------------------- #

_BYTES_TAG = "__b64__"
T = TypeVar("T")


def _json_default(o: Any) -> Any:
    if isinstance(o, (bytes, bytearray)):
        return {_BYTES_TAG: base64.b64encode(bytes(o)).decode("ascii")}
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def to_json(obj: Any) -> str:
    """Serialize a wire contract (Source / ASRResult / AgentEvent / TTSChunk) to JSON."""
    if isinstance(obj, TurnContext):
        raise TypeError(
            "TurnContext is an in-process control object (carries asyncio.Event) "
            "and is not JSON-serializable."
        )
    if not any(isinstance(obj, c) for c in WIRE_CONTRACTS):
        raise TypeError(f"{type(obj).__name__} is not a wire contract")
    return json.dumps(asdict(obj), default=_json_default, ensure_ascii=False)


def from_json(cls: type[T], data: str) -> T:
    """Reconstruct a wire contract instance of ``cls`` from its JSON string."""
    d = json.loads(data)
    if cls is Source:
        return Source(**d)  # type: ignore[return-value]
    if cls is ASRResult:
        return ASRResult(**d)  # type: ignore[return-value]
    if cls is AgentEvent:
        d = {**d, "citations": [Source(**c) for c in d.get("citations", [])]}
        return AgentEvent(**d)  # type: ignore[return-value]
    if cls is TTSChunk:
        ab = d.get("audio_bytes")
        if isinstance(ab, dict) and _BYTES_TAG in ab:
            d = {**d, "audio_bytes": base64.b64decode(ab[_BYTES_TAG])}
        return TTSChunk(**d)  # type: ignore[return-value]
    raise TypeError(f"Unknown contract type: {cls!r}")


# --------------------------------------------------------------------------- #
# Small convenience helpers (not part of the frozen shapes).
# --------------------------------------------------------------------------- #


def new_turn(session_id: str = "local", lang: str = "hi") -> TurnContext:
    """Create a fresh TurnContext with a unique turn_id and an unset cancel Event."""
    return TurnContext(
        turn_id=uuid.uuid4().hex[:12],
        session_id=session_id,
        lang=lang,
        cancel=asyncio.Event(),
        t_started=time.monotonic(),
    )
