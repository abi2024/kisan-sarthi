"""serving/metrics.py — per-leg latency / throughput feed for the live dashboard (L3.4).

Minimal in-memory implementation now so Lane 4's dashboard can build against the
interface; Nsight-driven per-leg profiling and the real emit path land in L3.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricsFeed:
    samples: list[tuple[str, float]] = field(default_factory=list)

    def emit(self, leg: str, ms: float) -> None:
        """Record a timing for one pipeline leg (asr / agent / llm / tts)."""
        self.samples.append((leg, ms))

    def latest(self, leg: str) -> float | None:
        for name, ms in reversed(self.samples):
            if name == leg:
                return ms
        return None
