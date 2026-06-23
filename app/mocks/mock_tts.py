"""Mock TTS — the Lane 2 -> Lane 1 seam, faked.

Consumes the agent's `AgentEvent` text-delta stream and yields one `TTSChunk` per
sentence (canned silent audio), all stamped with `turn_id`, terminating with
`is_last=True`. Aborts promptly if `ctx.cancel` is set (barge-in). The
`synthesize(events, ctx)` signature matches the real `StreamingTTS.synthesize`
(Milestone Map §5), so the real Lane-1 implementation is a drop-in replacement.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

from kisan_sarthi.contracts.models import AgentEvent, TTSChunk, TurnContext

_SENTENCE_END = re.compile(r"[.!?।]\s*$")  # includes the Devanagari danda (।)
_SAMPLE_RATE = 22050


def _fake_pcm(text: str, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Return silent 16-bit PCM roughly proportional to the sentence length.

    ~60 ms of audio per word — enough to be a non-empty, plausibly-sized payload
    without depending on a real vocoder.
    """
    words = max(1, len(text.split()))
    n_samples = int(sample_rate * 0.06 * words)
    return b"\x00\x00" * n_samples  # 2 bytes/sample, silence


class MockTTS:
    def __init__(self, chunk_delay_s: float = 0.02, sample_rate: int = _SAMPLE_RATE) -> None:
        self.chunk_delay_s = chunk_delay_s
        self.sample_rate = sample_rate

    async def synthesize(
        self, events: AsyncIterator[AgentEvent], ctx: TurnContext
    ) -> AsyncIterator[TTSChunk]:
        buffer = ""
        sentence_idx = 0

        async def emit(sentence: str, is_last: bool) -> TTSChunk:
            await asyncio.sleep(self.chunk_delay_s)
            return TTSChunk(
                turn_id=ctx.turn_id,
                audio_bytes=_fake_pcm(sentence, self.sample_rate),
                sample_rate=self.sample_rate,
                is_last=is_last,
                sentence_idx=sentence_idx,
            )

        async for ev in events:
            if ctx.cancel.is_set():
                return
            if ev.kind == "text" and ev.text_delta:
                buffer += ev.text_delta
                # Flush completed sentences as they form.
                while _SENTENCE_END.search(buffer):
                    sentence = buffer.strip()
                    buffer = ""
                    if sentence:
                        yield await emit(sentence, is_last=False)
                        sentence_idx += 1
            elif ev.kind in ("final", "escalate"):
                break

        if ctx.cancel.is_set():
            return
        # Flush any trailing partial sentence as the final chunk.
        tail = buffer.strip()
        yield await emit(tail or " ", is_last=True)
