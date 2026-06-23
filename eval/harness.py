"""eval/harness.py — the single evaluation harness (L4.3).

Runs every suite that has inputs available — ASR WER, RAG groundedness, intent-routing accuracy,
end-to-end latency — and writes one `eval/results/report.md`: the single source of truth for every
number in the pitch. Suites with no data yet are reported as PENDING rather than failing, so the
harness runs today against the sample fixtures and fills in as each lane delivers real held-out
sets.

    python -m eval.harness
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from eval.metrics.groundedness import mean_groundedness
from eval.metrics.latency import load_bench_json
from eval.metrics.wer import corpus_wer

DATA_DIR = Path("eval/data/samples")
RESULTS_DIR = Path("eval/results")
ASR_EVAL = DATA_DIR / "asr_eval.jsonl"
GROUNDING_EVAL = DATA_DIR / "grounding_eval.jsonl"
ROUTING_EVAL = DATA_DIR / "routing_eval.jsonl"
BENCH_JSON = RESULTS_DIR / "baseline.json"


@dataclass
class Suite:
    name: str
    status: str
    summary: str
    detail: dict


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _asr_suite() -> Suite:
    rows = _read_jsonl(ASR_EVAL)
    if not rows:
        return Suite("ASR WER", "pending", "no clips/transcripts yet (Lane 1)", {})
    score = corpus_wer((r["ref"], r["hyp"]) for r in rows)
    return Suite(
        "ASR WER",
        "ok",
        f"corpus WER = {score:.3f} over {len(rows)} utts",
        {"wer": round(score, 4), "n": len(rows)},
    )


def _grounding_suite() -> Suite:
    rows = _read_jsonl(GROUNDING_EVAL)
    if not rows:
        return Suite("RAG groundedness", "pending", "no answers/sources yet (Lane 2)", {})
    score = mean_groundedness(rows)
    return Suite(
        "RAG groundedness",
        "ok",
        f"mean groundedness = {score:.3f} over {len(rows)} answers",
        {"groundedness": round(score, 4), "n": len(rows)},
    )


def _routing_suite() -> Suite:
    rows = _read_jsonl(ROUTING_EVAL)
    if not rows:
        return Suite("Intent routing", "pending", "no routing labels yet (Lane 2)", {})
    correct = sum(1 for r in rows if r["pred"] == r["gold"])
    acc = correct / len(rows)
    return Suite(
        "Intent routing",
        "ok",
        f"accuracy = {acc:.3f} ({correct}/{len(rows)})",
        {"accuracy": round(acc, 4), "n": len(rows)},
    )


def _latency_suite() -> Suite:
    bench = load_bench_json(BENCH_JSON)
    if not bench:
        return Suite(
            "Serving latency", "pending", "no baseline.json yet (Lane 3 / capture-baseline)", {}
        )
    ttft = bench.get("ttft_ms", {})
    total = bench.get("total_latency_ms", {})
    tps = bench.get("tokens_per_sec", {})
    summary = (
        f"TTFT p50/p95 = {ttft.get('p50')}/{ttft.get('p95')} ms · "
        f"total p50/p95 = {total.get('p50')}/{total.get('p95')} ms · "
        f"tok/s(agg) = {tps.get('aggregate')}  [label={bench.get('label')}]"
    )
    return Suite(
        "Serving latency",
        "ok",
        summary,
        {"ttft_ms": ttft, "total_latency_ms": total, "tokens_per_sec": tps},
    )


class Harness:
    def run_all(self) -> dict:
        suites = [_asr_suite(), _grounding_suite(), _routing_suite(), _latency_suite()]
        report = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "suites": {
                s.name: {"status": s.status, "summary": s.summary, "detail": s.detail}
                for s in suites
            },
        }
        self._write_markdown(suites, report["generated"])
        return report

    def _write_markdown(self, suites: list[Suite], generated: str) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Kisan Sarthi — Evaluation Report",
            "",
            f"_Generated {generated}_",
            "",
            "Single source of truth for the numbers in the pitch. PENDING suites fill in as each "
            "lane delivers real held-out data.",
            "",
            "| Suite | Status | Result |",
            "| ----- | ------ | ------ |",
        ]
        for s in suites:
            badge = "✅ ok" if s.status == "ok" else "⏳ pending"
            lines.append(f"| {s.name} | {badge} | {s.summary} |")
        lines.append("")
        (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = Harness().run_all()
    for name, s in report["suites"].items():
        flag = "ok " if s["status"] == "ok" else "PEND"
        print(f"[{flag}] {name}: {s['summary']}")
    print(f"\n[harness] wrote {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
