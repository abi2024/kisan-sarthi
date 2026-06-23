"""serving/trtllm/build_engine.py — compile a TensorRT-LLM engine. STUB (Wk2, L3.2).

TRT-LLM needs an offline engine build per model/precision/GPU before serving. That build
step lives here; the served endpoint stays the identical OpenAI-compatible API, so moving
from vLLM to TRT-LLM is invisible to Lane 2. Fallback if TRT-LLM is immature on the box:
vLLM FP8 (still a real speedup) — see risk register.
"""

from __future__ import annotations

from pathlib import Path


def build_engine(checkpoint: str | Path, precision: str) -> Path:
    """Produce a TRT-LLM engine for `checkpoint` at `precision`; return the engine dir."""
    raise NotImplementedError(
        "L3.2: build the TRT-LLM engine (FP8) with TensorRT Model Optimizer, then serve "
        "via trtllm-serve behind the same /v1/chat/completions endpoint."
    )
