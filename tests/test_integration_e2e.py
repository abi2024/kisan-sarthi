"""Level-3 integration test (runs in CI, no GPU, no live port).

Wires the components we have to mocks for the rest and asserts a turn completes
end-to-end through the REAL OpenAI wire format. The mock LLM is mounted in-process via an
httpx ASGI transport injected into the OpenAI client, so this exercises serving/client.py
and the real protocol without binding a port. Catches "the pieces don't compose" within
minutes of a merge — not in Week 3.
"""

import httpx
import pytest

from app.main import run_turn
from app.mocks.mock_asr import MockASR
from app.mocks.mock_llm import app as mock_llm_app
from app.mocks.mock_tts import MockTTS
from kisan_sarthi.contracts.models import AgentEvent, new_turn
from kisan_sarthi.serving.client import make_client, stream_completion

pytestmark = pytest.mark.integration


def _asgi_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_llm_app), base_url="http://mock"
    )


async def test_client_streams_text_against_mock_llm_over_asgi():
    client = make_client("http://mock", http_client=_asgi_http_client())
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
    """mock ASR -> agent-glue(LLM client -> mock LLM) -> mock TTS, sharing one TurnContext."""
    ctx = new_turn(session_id="ci", lang="hi")
    asr = MockASR(chunk_delay_s=0.0)
    tts = MockTTS(chunk_delay_s=0.0)

    # Patch the agent glue's client construction to use the in-process ASGI transport by
    # running the turn against a make_client bound to the mounted app.
    import app.main as main_mod

    orig_stream = main_mod.stream_completion

    def patched_stream(messages, **kwargs):
        kwargs.pop("origin", None)
        client = make_client("http://mock", http_client=_asgi_http_client())
        return orig_stream(messages, client=client, **kwargs)

    main_mod.stream_completion = patched_stream
    try:
        summary = await run_turn(asr, tts, ctx, origin="http://mock")
    finally:
        main_mod.stream_completion = orig_stream

    assert summary["turn_id"] == ctx.turn_id
    assert summary["transcript"], "ASR produced no final transcript"
    assert summary["tts_chunks"] >= 1
    assert summary["audio_bytes"] > 0
    assert summary["timing_ms"]["total_ms"] > 0


async def test_barge_in_cancels_the_turn():
    """Setting ctx.cancel stops LLM streaming and TTS for that turn."""
    ctx = new_turn()
    ctx.cancel.set()  # barge-in before generation
    client = make_client("http://mock", http_client=_asgi_http_client())
    events = []
    async for delta in stream_completion(
        [{"role": "user", "content": "long answer"}], client=client, cancel=ctx.cancel
    ):
        events.append(AgentEvent(turn_id=ctx.turn_id, kind="text", text_delta=delta))
    await client.close()
    assert events == [], "cancel did not stop LLM streaming"
