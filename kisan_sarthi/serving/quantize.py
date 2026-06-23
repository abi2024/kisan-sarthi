"""serving/quantize.py — quantization (Wk2 FP8 / Wk3 NVFP4). STUB.

Filled in at L3.2 (FP8 via TensorRT Model Optimizer, ~99% accuracy retention vs FP16)
and L3.3 (NVFP4 4-bit, ~21 GB MoE footprint). Signatures are fixed now so the rest of
the lane and the benchmark naming can be built against them.
"""

from __future__ import annotations

from pathlib import Path


def quantize(checkpoint: str | Path, precision: str) -> Path:
    """Quantize `checkpoint` to `precision` ('fp8' | 'nvfp4'); return the output path."""
    raise NotImplementedError(
        "L3.2/L3.3: implement ModelOpt FP8 (Wk2) then NVFP4 (Wk3). "
        "Pin nvidia-modelopt >= 0.17 (risk register)."
    )


def verify_quality(quantized: str | Path, eval_set: str | Path) -> float:
    """Return accuracy retention vs the FP16 baseline (target ~0.99). STUB."""
    raise NotImplementedError("L3.2: confirm quality held before claiming any speedup.")
