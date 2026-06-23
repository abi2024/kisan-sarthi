"""app/main.py — the system entrypoint and end-to-end demo driver.

Gate-0 form: build a Session and drive one turn through it with mocks (mock ASR -> LLM client
-> mock TTS), printing per-leg timing. The turn-wiring lives in app/session.py (the L4.2 spine);
this file just composes the components and launches. Real Lane-1 speech and Lane-3 serving drop
in by swapping the mocks and pointing PIPELINE=real.

    PIPELINE=all-mock python -m app.main      # laptop: every seam mocked (needs mock LLM up)
    PIPELINE=real     python -m app.main      # box: real components (Wk2+)
"""

from __future__ import annotations

import asyncio
import os

from app.session import Session


def _build_pipeline(pipeline: str):
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
    asr, tts = _build_pipeline(pipeline)
    session = Session(asr, tts, session_id="demo", lang="hi", origin=origin)
    ctx = session.new_turn_ctx()

    print(f"[e2e] PIPELINE={pipeline} endpoint={origin} turn_id={ctx.turn_id}", flush=True)
    result = await session.run_turn(ctx)

    print(f"\n  transcript : {result.transcript}")
    print(f"  tts chunks : {result.tts_chunks}  ({result.audio_bytes} audio bytes)")
    print(f"  cancelled  : {result.cancelled}")
    print("  timing (ms):")
    for leg, ms in result.timing_ms.items():
        print(f"     {leg:<16} {ms}")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
