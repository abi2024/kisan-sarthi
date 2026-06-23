# Kisan Sarthi — Evaluation Report

_Generated 2026-06-23T20:11:45.882315+00:00_

Single source of truth for the numbers in the pitch. PENDING suites fill in as each lane delivers real held-out data.

| Suite | Status | Result |
| ----- | ------ | ------ |
| ASR WER | ✅ ok | corpus WER = 0.080 over 3 utts |
| RAG groundedness | ✅ ok | mean groundedness = 0.667 over 3 answers |
| Intent routing | ✅ ok | accuracy = 0.800 (4/5) |
| Serving latency | ✅ ok | TTFT p50/p95 = 4.86/543.03 ms · total p50/p95 = 13.79/551.6 ms · tok/s(agg) = 263.56  [label=baseline] |

