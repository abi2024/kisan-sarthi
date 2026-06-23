"""Level-2 contract-conformance tests for the mocks.

Every mock must conform to the same dataclass / wire shapes as the real thing, so it can
be swapped for the real component invisibly. Also checks first-class cancellation:
setting `ctx.cancel` stops the ASR/TTS streams. The mock LLM is checked against the
OpenAI HTTP shape via an in-process ASGI client (no port, CI-friendly).
"""

import httpx
import pytest

from app.mocks.mock_asr import MockASR
from app.mocks.mock_llm import app as mock_llm_app
from app.mocks.mock_tts import MockTTS
from kisan_sarthi.contracts.models import AgentEvent, ASRResult, TTSChunk, new_turn

pytestmark = pytest.mark.contract


# --------------------------- mock ASR --------------------------------------- #


async def test_mock_asr_yields_asrresults_with_turn_id_and_final():
    ctx = new_turn()
    results = [r async for r in MockASR(chunk_delay_s=0.0).stream(ctx)]
    assert results, "mock ASR yielded nothing"
    assert all(isinstance(r, ASRResult) for r in results)
    assert all(r.turn_id == ctx.turn_id for r in results)
    assert results[-1].is_final and not any(r.is_final for r in results[:-1])


async def test_mock_asr_respects_cancel():
    ctx = new_turn()
    ctx.cancel.set()  # barge-in before we start
    results = [r async for r in MockASR(chunk_delay_s=0.0).stream(ctx)]
    assert results == [], "mock ASR ignored ctx.cancel"


# --------------------------- mock TTS --------------------------------------- #


async def _events(ctx, *deltas):
    for d in deltas:
        yield AgentEvent(turn_id=ctx.turn_id, kind="text", text_delta=d)
    yield AgentEvent(turn_id=ctx.turn_id, kind="final")


async def test_mock_tts_yields_chunks_with_turn_id_and_islast():
    ctx = new_turn()
    chunks = [
        c
        async for c in MockTTS(chunk_delay_s=0.0).synthesize(
            _events(ctx, "Pehla vakya. ", "Doosra vakya."), ctx
        )
    ]
    assert chunks, "mock TTS yielded nothing"
    assert all(isinstance(c, TTSChunk) for c in chunks)
    assert all(c.turn_id == ctx.turn_id for c in chunks)
    assert all(len(c.audio_bytes) > 0 for c in chunks)
    assert chunks[-1].is_last and not any(c.is_last for c in chunks[:-1])


async def test_mock_tts_respects_cancel():
    ctx = new_turn()

    async def events_then_cancel():
        yield AgentEvent(turn_id=ctx.turn_id, kind="text", text_delta="Ek vakya. ")
        ctx.cancel.set()  # barge-in mid-stream
        yield AgentEvent(turn_id=ctx.turn_id, kind="text", text_delta="Doosra. ")

    chunks = [c async for c in MockTTS(chunk_delay_s=0.0).synthesize(events_then_cancel(), ctx)]
    assert all(c.turn_id == ctx.turn_id for c in chunks)
    # Cancellation must halt synthesis (no terminal is_last chunk forced out).
    assert not any(c.is_last for c in chunks)


# --------------------------- mock LLM (OpenAI shape) ------------------------ #


def _asgi_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=mock_llm_app)
    return httpx.AsyncClient(transport=transport, base_url="http://mock")


async def test_mock_llm_health_returns_200():
    async with _asgi_client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_mock_llm_nonstreaming_openai_shape():
    async with _asgi_client() as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-llm",
                "messages": [{"role": "user", "content": "namaste"}],
                "max_tokens": 16,
                "reasoning": False,  # tolerated extra_body field
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"]
    assert body["usage"]["completion_tokens"] >= 1


async def test_mock_llm_streaming_sse_terminates_with_done():
    async with _asgi_client() as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "mock-llm",
                "messages": [{"role": "user", "content": "namaste"}],
                "stream": True,
                "max_tokens": 8,
            },
        ) as resp:
            assert resp.status_code == 200
            lines = [ln async for ln in resp.aiter_lines()]
    data_lines = [ln[len("data: ") :] for ln in lines if ln.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"
    assert len(data_lines) >= 3  # role + >=1 content + terminal
