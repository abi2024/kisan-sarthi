"""app/session.py — per-call session state and the turn loop (L4.2 spine).

A `Session` owns a call: it mints a fresh `TurnContext` per turn, drives
ASR-stream -> agent -> TTS-stream sharing that context, tracks per-leg timing, and exposes
`barge_in()` to cancel the active turn. It is the seam the real pipeline drops into: the
`agent` is any callable matching Lane 2's real `AgentGraph.run` signature
`(ctx, asr_stream) -> AsyncIterator[AgentEvent]`, so swapping the placeholder for the real
agent is a one-line change.

Build/test this entirely against mocks (no GPU). The default agent is an LLM passthrough that
streams from whatever OpenAI-compatible endpoint `origin` points at (the mock LLM today, real
vLLM on the box).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from kisan_sarthi.contracts.models import AgentEvent, ASRResult, TurnContext, new_turn
from kisan_sarthi.serving.client import stream_completion

# An agent consumes the ASR stream for a turn and streams AgentEvents back.
# This is exactly Lane 2's AgentGraph.run signature.
AgentFn = Callable[[TurnContext, AsyncIterator[ASRResult]], AsyncIterator[AgentEvent]]


@dataclass
class TurnResult:
    turn_id: str
    transcript: str
    tts_chunks: int
    audio_bytes: int
    cancelled: bool
    timing_ms: dict[str, float] = field(default_factory=dict)


def make_llm_passthrough_agent(origin: str | None = None) -> AgentFn:
    """Default placeholder agent: take the final ASR transcript, stream the LLM reply as
    text AgentEvents, end with a `final` marker. Lane 2's AgentGraph.run replaces this."""

    async def agent(ctx: TurnContext, asr: AsyncIterator[ASRResult]) -> AsyncIterator[AgentEvent]:
        final_text = ""
        async for r in asr:
            if r.is_final:
                final_text = r.text
        async for delta in stream_completion(
            [{"role": "user", "content": final_text}], origin=origin, cancel=ctx.cancel
        ):
            yield AgentEvent(turn_id=ctx.turn_id, kind="text", text_delta=delta)
        yield AgentEvent(turn_id=ctx.turn_id, kind="final")

    return agent


class Session:
    """One voice call. Holds the components and runs turns; barge-in cancels the live turn."""

    def __init__(
        self,
        asr,
        tts,
        *,
        agent: AgentFn | None = None,
        session_id: str = "local",
        lang: str = "hi",
        origin: str | None = None,
    ) -> None:
        self.asr = asr
        self.tts = tts
        self.agent: AgentFn = agent or make_llm_passthrough_agent(origin)
        self.session_id = session_id
        self.lang = lang
        self._active: TurnContext | None = None
        self.turns: list[TurnResult] = []

    def new_turn_ctx(self) -> TurnContext:
        ctx = new_turn(session_id=self.session_id, lang=self.lang)
        self._active = ctx
        return ctx

    def barge_in(self) -> bool:
        """Cancel the currently-active turn (sets ctx.cancel). Returns True if one was live."""
        if self._active is not None and not self._active.cancel.is_set():
            self._active.cancel.set()
            return True
        return False

    async def run_turn(self, ctx: TurnContext | None = None) -> TurnResult:
        """Drive one full turn end-to-end, sharing `ctx` across ASR, agent, and TTS."""
        ctx = ctx or self.new_turn_ctx()
        t0 = time.monotonic()
        timing: dict[str, float] = {}
        captured = {"transcript": ""}

        # Observe the ASR stream as it flows to the agent: capture the final transcript and
        # the asr-final timestamp without breaking the agent's (ctx, asr_stream) contract.
        async def observed_asr() -> AsyncIterator[ASRResult]:
            async for r in self.asr.stream(ctx):
                if r.is_final:
                    captured["transcript"] = r.text
                    timing.setdefault("asr_final_ms", (time.monotonic() - t0) * 1000.0)
                yield r

        events = self.agent(ctx, observed_asr())

        first_audio = True
        n_chunks = 0
        n_bytes = 0
        async for chunk in self.tts.synthesize(events, ctx):
            if first_audio:
                timing["first_audio_ms"] = (time.monotonic() - t0) * 1000.0
                first_audio = False
            assert chunk.turn_id == ctx.turn_id, "TTSChunk carried the wrong turn_id"
            n_chunks += 1
            n_bytes += len(chunk.audio_bytes)

        timing["total_ms"] = (time.monotonic() - t0) * 1000.0
        result = TurnResult(
            turn_id=ctx.turn_id,
            transcript=captured["transcript"],
            tts_chunks=n_chunks,
            audio_bytes=n_bytes,
            cancelled=ctx.cancel.is_set(),
            timing_ms={k: round(v, 1) for k, v in timing.items()},
        )
        self.turns.append(result)
        if self._active is ctx:
            self._active = None
        return result

    async def converse(self, n_turns: int = 1) -> list[TurnResult]:
        """Run `n_turns` sequential turns, each with its own fresh TurnContext."""
        return [await self.run_turn() for _ in range(n_turns)]  