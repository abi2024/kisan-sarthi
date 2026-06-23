"""app/main.py — the system entrypoint and end-to-end driver.

Gate 0 form: wire the WHOLE SHAPE of a turn with mocks — mock ASR -> LLM client (against
a running mock LLM or the real server) -> mock TTS — and print the per-leg timing. This
is the "always integrated, increasingly real" proof: from day 1 we run the full pipeline
shape, then replace mocks with real components one at a time (the real Lane-1 speech and
Lane-3 serving land in Wk1-2; this driver grows into L4.2's full wiring).

    PIPELINE=all-mock python -m app.main      # laptop: every seam mocked (needs mock LLM up)
    PIPELINE=real     python -m app.main      # box: real components (Wk2+)

The Makefile / CI boots the mock LLM; this driver just connects to KISAN_SERVING_URL.

Lane-4 note: the small "agent" glue here (wrapping LLM text deltas as AgentEvents) is
placeholder wiring at the seam. Lane 2's real AgentGraph.run replaces it.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator

from kisan_sarthi.contracts.models import AgentEvent, TurnContext, new_turn
from kisan_sarthi.serving.client import stream_completion


async def _agent_glue(text: str, ctx: TurnContext, origin: str) -> AsyncIterator[AgentEvent]:
    """Placeholder agent: forward LLM text deltas as AgentEvents, end with a final marker.

    Lane 2's AgentGraph.run (intent -> RAG -> tools -> guardrails) replaces this.
    """
    async for delta in stream_completion(
        [{"role": "user", "content": text}], origin=origin, cancel=ctx.cancel
    ):
        yield AgentEvent(turn_id=ctx.turn_id, kind="text", text_delta=delta)
    yield AgentEvent(turn_id=ctx.turn_id, kind="final")


async def run_turn(asr, tts, ctx: TurnContext, *, origin: str) -> dict:
    """Drive one turn end-to-end; return a timing + content summary."""
    t0 = time.monotonic()
    legs: dict[str, float] = {}

    # 1) ASR: consume the stream, keep the final transcript.
    final_text = ""
    async for r in asr.stream(ctx):
        if r.is_final:
            final_text = r.text
            legs["asr_final_ms"] = (time.monotonic() - t0) * 1000.0

    # 2) Agent + LLM + 3) TTS, overlapped: TTS consumes agent events as they stream.
    events = _agent_glue(final_text, ctx, origin)
    first_audio_ms: float | None = None
    full_audio_bytes = 0
    n_chunks = 0
    async for chunk in tts.synthesize(events, ctx):
        if first_audio_ms is None:
            first_audio_ms = (time.monotonic() - t0) * 1000.0
        full_audio_bytes += len(chunk.audio_bytes)
        n_chunks += 1
        assert chunk.turn_id == ctx.turn_id, "TTSChunk carried the wrong turn_id"

    legs["first_audio_ms"] = first_audio_ms if first_audio_ms is not None else float("nan")
    legs["total_ms"] = (time.monotonic() - t0) * 1000.0
    return {
        "turn_id": ctx.turn_id,
        "transcript": final_text,
        "tts_chunks": n_chunks,
        "audio_bytes": full_audio_bytes,
        "timing_ms": {k: round(v, 1) for k, v in legs.items()},
    }


def _build_pipeline(pipeline: str, origin: str):
    if pipeline == "all-mock":
        from app.mocks.mock_asr import MockASR
        from app.mocks.mock_tts import MockTTS

        return MockASR(), MockTTS()
    if pipeline == "real":
        raise SystemExit(
            "PIPELINE=real needs Lane-1 speech (real ASR/TTS) and Lane-3 real serving, "
            "which land in Wk1-2. Use PIPELINE=all-mock for Gate 0. (Mixed real+mock is "
            "supported once real components exist.)"
        )
    raise SystemExit(f"Unknown PIPELINE='{pipeline}'. Use all-mock | real.")


async def _amain() -> None:
    pipeline = os.getenv("PIPELINE", "all-mock")
    origin = os.getenv("KISAN_SERVING_URL", "http://localhost:8000")
    asr, tts = _build_pipeline(pipeline, origin)
    ctx = new_turn(session_id="demo", lang="hi")

    print(f"[e2e] PIPELINE={pipeline} endpoint={origin} turn_id={ctx.turn_id}", flush=True)
    summary = await run_turn(asr, tts, ctx, origin=origin)

    print(f"\n  transcript : {summary['transcript']}")
    print(f"  tts chunks : {summary['tts_chunks']}  ({summary['audio_bytes']} audio bytes)")
    print("  timing (ms):")
    for leg, ms in summary["timing_ms"].items():
        print(f"     {leg:<16} {ms}")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
