"""Mock LLM server — the OpenAI-compatible seam, faked.

This is a REAL HTTP server that speaks the *exact* same wire protocol as the real
serving layer (vLLM / TensorRT-LLM): `GET /health` and `POST /v1/chat/completions`
with streaming SSE in OpenAI's chunk format. Because the seam is the HTTP protocol
(not a dataclass), Lane 2 points its `openai` client at this mock and CANNOT tell it
apart from the real server — swapping mock <-> real is a base-url change, nothing else.

It is built here, in Lane 3, *with* the real server (serving/server.py) so the two
shapes cannot drift (TESTING.md §5). It returns canned tokens with a small per-token
delay so that bench.py measures a meaningful TTFT and tokens/sec.

Run it:
    uvicorn app.mocks.mock_llm:app --host 127.0.0.1 --port 8000
    # or
    python -m app.mocks.mock_llm --port 8000

Env:
    MOCK_LLM_TOKEN_DELAY_S   per-token delay in seconds (default 0.01)
    MOCK_LLM_MODEL_NAME      model name echoed in responses (default 'mock-llm')
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

TOKEN_DELAY_S = float(os.getenv("MOCK_LLM_TOKEN_DELAY_S", "0.01"))
MODEL_NAME = os.getenv("MOCK_LLM_MODEL_NAME", "mock-llm")

# A canned, domain-flavoured reply. Whitespace-tokenised to simulate token streaming.
# (A mock returns the same shape regardless of prompt — content is not the point.)
CANNED_REPLY = (
    "PMFBY ke tahat fasal nuksan ki suchna 72 ghante ke andar deni hoti hai. "
    "Aap apne nazdeeki bank ya CSC se sampark karein. "
    "Yeh jaankari aam taur par hai, kripya official source se confirm karein."
)
_CANNED_TOKENS = CANNED_REPLY.split(" ")

app = FastAPI(title="Kisan Sarthi — Mock LLM (OpenAI-compatible)")


def _chat_id() -> str:
    return "chatcmpl-mock-" + uuid.uuid4().hex[:16]


def _select_tokens(max_tokens: int | None) -> list[str]:
    toks = _CANNED_TOKENS
    if max_tokens is not None and max_tokens > 0:
        toks = toks[:max_tokens]
    # Re-attach spaces between tokens so the reassembled string reads naturally.
    return [(t if i == 0 else " " + t) for i, t in enumerate(toks)]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # Read the body as raw JSON so unknown extra_body fields (e.g. `reasoning`) are
    # tolerated exactly as a permissive real server would tolerate them.
    body = await request.json()
    model = body.get("model", MODEL_NAME)
    stream = bool(body.get("stream", False))
    max_tokens = body.get("max_tokens")
    include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))

    tokens = _select_tokens(max_tokens if isinstance(max_tokens, int) else None)
    # Rough prompt-token estimate from the messages, for a plausible usage block.
    prompt_text = " ".join(
        str(m.get("content", "")) for m in body.get("messages", []) if isinstance(m, dict)
    )
    prompt_tokens = max(1, len(prompt_text.split()))
    completion_tokens = len(tokens)
    cid = _chat_id()
    created = int(time.time())

    if not stream:
        content = "".join(tokens)
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    async def event_stream() -> AsyncIterator[bytes]:
        # 1) role delta
        first = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n".encode()

        # 2) content deltas, one token at a time with a small delay
        for tok in tokens:
            await asyncio.sleep(TOKEN_DELAY_S)
            chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

        # 3) terminal chunk with finish_reason
        last = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        if include_usage:
            last["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        yield f"data: {json.dumps(last)}\n\n".encode()

        # 4) OpenAI sentinel
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Kisan Sarthi mock LLM (OpenAI-compatible)")
    parser.add_argument("--host", default=os.getenv("KISAN_SERVING_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("KISAN_SERVING_PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
