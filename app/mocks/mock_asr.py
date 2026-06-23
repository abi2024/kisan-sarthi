"""Mock ASR — the Lane 1 -> Lane 2 seam, faked.

An in-process async generator that yields canned `ASRResult`s (a few interim, then a
final), all stamped with the turn's `turn_id`, and stops early if `ctx.cancel` is set.
Its `stream(ctx)` signature matches the real `StreamingASR.stream` (Milestone Map §5),
so the real Lane-1 implementation is a drop-in replacement.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from kisan_sarthi.contracts.models import ASRResult, TurnContext

# Canned transcript, revealed incrementally to imitate streaming recognition.
_DEFAULT_FINAL = "aaj Indore mandi mein soybean ka bhaav kya hai?"


class MockASR:
    def __init__(self, final_text: str = _DEFAULT_FINAL, chunk_delay_s: float = 0.05) -> None:
        self.final_text = final_text
        self.chunk_delay_s = chunk_delay_s

    async def stream(self, ctx: TurnContext) -> AsyncIterator[ASRResult]:
        words = self.final_text.split(" ")
        # Emit a couple of interim hypotheses, then the final.
        cutpoints = sorted({max(1, len(words) // 3), max(2, 2 * len(words) // 3)})
        t_start = time.monotonic() - ctx.t_started
        for cut in cutpoints:
            if ctx.cancel.is_set():
                return
            await asyncio.sleep(self.chunk_delay_s)
            yield ASRResult(
                turn_id=ctx.turn_id,
                text=" ".join(words[:cut]),
                is_final=False,
                lang=ctx.lang,
                t_start=t_start,
                t_end=time.monotonic() - ctx.t_started,
                confidence=0.6,
            )
        if ctx.cancel.is_set():
            return
        await asyncio.sleep(self.chunk_delay_s)
        yield ASRResult(
            turn_id=ctx.turn_id,
            text=self.final_text,
            is_final=True,
            lang=ctx.lang,
            t_start=t_start,
            t_end=time.monotonic() - ctx.t_started,
            confidence=0.92,
        )

    async def push_audio(self, chunk: bytes) -> None:  # parity with real signature
        """No-op for the mock; the real StreamingASR feeds PCM into its decode queue."""
        return None
