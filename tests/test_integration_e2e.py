"""Level-3 integration test (runs in CI, no GPU, no live port).

Drives a full turn through the Session with the mock ASR/TTS and an agent bound to the mock LLM
mounted in-process via an httpx ASGI transport (real OpenAI wire format, no port). Catches "the
pieces don't compose" within minutes of a merge, not in Week 3.
"""

import httpx
import pytest

from app.mocks.mock_asr import MockASR
from app.mocks.mock_llm import app as mock_llm_app
from app.mocks.mock_tts import MockTTS
from app.session import AgentFn, Session
from kisan_sarthi.contracts.models import AgentEvent
from kisan_sarthi.serving.client import make_client, stream_completion

pytestmark = pytest.mark.integration


def _asgi_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_llm_app), base_url="http://mock"
    )


def _mock_llm_agent() -> AgentFn:
    """An agent matching AgentGraph.run's signature, wired to the in-process mock LLM."""

    async def agent(ctx, asr):
        final = ""
        async for r in asr:
            if r.is_final:
                final = r.text
        client = make_client("http://mock", http_client=_asgi_client())
        try:
            async for delta in stream_completion(
                [{"role": "user", "content": final}], client=client, cancel=ctx.cancel
            ):
                yield AgentEvent(turn_id=ctx.turn_id, kind="text", text_delta=delta)
        finally:
            await client.close()
        yield AgentEvent(turn_id=ctx.turn_id, kind="final")

    return agent


async def test_client_streams_text_against_mock_llm_over_asgi():
    client = make_client("http://mock", http_client=_asgi_client())
    deltas = [
        d
        async for d in stream_completion(
            [{"role": "user", "content": "PMFBY claim window?"}], client=client, max_tokens=12
        )
    ]
    await client.close()
    assert deltas, "no text deltas streamed from mock LLM"
    assert "".join(deltas).strip()


async def test_full_turn_shape_completes_with_mocks():
    """mock ASR -> agent(LLM client -> mock LLM) -> mock TTS, sharing one TurnContext."""
    session = Session(
        MockASR(chunk_delay_s=0.0), MockTTS(chunk_delay_s=0.0), agent=_mock_llm_agent()
    )
    result = await session.run_turn()
    assert result.transcript, "ASR produced no final transcript"
    assert result.tts_chunks >= 1
    assert result.audio_bytes > 0
    assert result.cancelled is False
    assert result.timing_ms["total_ms"] > 0


async def test_barge_in_cancels_the_turn():
    """Setting ctx.cancel before generation stops the turn with no audio."""
    session = Session(
        MockASR(chunk_delay_s=0.0), MockTTS(chunk_delay_s=0.0), agent=_mock_llm_agent()
    )
    ctx = session.new_turn_ctx()
    ctx.cancel.set()
    result = await session.run_turn(ctx)
    assert result.cancelled is True
    assert result.tts_chunks == 0


async def test_multi_turn_session_gives_distinct_turn_ids():
    session = Session(
        MockASR(chunk_delay_s=0.0), MockTTS(chunk_delay_s=0.0), agent=_mock_llm_agent()
    )
    results = await session.converse(n_turns=2)
    assert len(results) == 2
    assert results[0].turn_id != results[1].turn_id
    assert all(r.transcript for r in results)
    assert len(session.turns) == 2
