"""serving/client.py — the OpenAI-compatible client Lane 2's agent calls.

The agent never writes bespoke HTTP code: it imports `make_client` / `stream_completion`
from here, which wrap the stock `openai` async SDK pointed at our serving endpoint. The
SAME functions work against the mock LLM and the real server (vLLM / TRT-LLM) because
both expose the identical `/v1/chat/completions` shape.

Cancellation: pass a `TurnContext.cancel` event; when barge-in sets it, the stream is
closed (HTTP request aborted) and generation stops — this is how `ctx.cancel` reaches
the LLM.

Env:
    KISAN_SERVING_URL   server origin, no trailing /v1 (default http://localhost:8000)
    KISAN_MODEL_ID      model name sent in the request (default 'mock-llm')
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

DEFAULT_ORIGIN = os.getenv("KISAN_SERVING_URL", "http://localhost:8000")
DEFAULT_MODEL = os.getenv("KISAN_MODEL_ID", "mock-llm")


def base_url_for(origin: str | None = None) -> str:
    """OpenAI SDK base_url (the server origin with the /v1 prefix appended)."""
    return f"{(origin or DEFAULT_ORIGIN).rstrip('/')}/v1"


def make_client(
    origin: str | None = None,
    *,
    api_key: str = "not-needed",
    http_client: Any | None = None,
    timeout: float = 30.0,
) -> AsyncOpenAI:
    """Build an AsyncOpenAI client for our endpoint.

    `http_client` lets tests inject an httpx ASGITransport so the mock app can be
    exercised in-process (no real port) while still going through the real wire format.
    """
    kwargs: dict[str, Any] = {
        "base_url": base_url_for(origin),
        "api_key": api_key,
        "timeout": timeout,
    }
    if http_client is not None:
        kwargs["http_client"] = http_client
    return AsyncOpenAI(**kwargs)


async def stream_completion(
    messages: list[dict[str, str]],
    *,
    client: AsyncOpenAI | None = None,
    origin: str | None = None,
    model: str | None = None,
    max_tokens: int = 256,
    reasoning: bool = False,
    cancel: asyncio.Event | None = None,
) -> AsyncIterator[str]:
    """Stream assistant text deltas for `messages`. Stops if `cancel` is set.

    `reasoning` toggles the model's reasoning trace. Nemotron-3-Nano controls this via
    `chat_template_kwargs={"enable_thinking": ...}` (reasoning is ON by default); the voice
    path wants it OFF for latency. The mock ignores the field, so this is safe everywhere.
    """
    client = client or make_client(origin)
    stream = await client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,  # type: ignore[arg-type]
        stream=True,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": reasoning}},
    )
    try:
        async for chunk in stream:
            if cancel is not None and cancel.is_set():
                break
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
    finally:
        await stream.close()  # abort the HTTP request (barge-in / early exit)
