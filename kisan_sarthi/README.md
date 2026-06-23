# Lane 4 — Integration, Demo & Eval · Deliverables

**Owner:** Integration engineer · **Owns:** `app/`, `eval/`, `.github/`
**Mandate:** the glue and the proof — CI, the end-to-end wiring, the evaluation harness, the live
latency dashboard, the demo app, and the fallback video. The lane that makes the other three add
up to a working product and proves it with numbers.

**Contracts you orchestrate** (frozen — see `kisan_sarthi/contracts/models.py`): you mint the
`TurnContext` (Lane 4 → all) and wire the seams together — ASR stream → `AgentGraph.run` → TTS
stream, all sharing one `ctx`, with barge-in setting `ctx.cancel`. You don't implement the lanes;
you compose them at the seams and keep them honest.

**State now:** L4.1 is **done** (CI + mocks green) and L4.2 is **seeded** in `app/main.py` (the
Gate-0 all-mock driver — `run_turn` already wires ASR → agent-glue → TTS with per-leg timing).
Your job is to grow that seed into the real composition and build the eval + demo surfaces.

**Definition of success:** a single command boots the whole system; a judge runs all four use
cases live; the dashboard shows the latency story in real time; the eval harness produces the
WER / groundedness / routing / latency numbers behind every claim; and a recorded fallback
guarantees the demo survives a network failure.

## Milestones

| ID | Wk | Deliverable | Key files / signatures | ✅ Validation gate (what "done" means) |
|----|----|-------------|------------------------|----------------------------------------|
| **L4.1** ✅ | 1 | CI (lint/unit/integration) + contract-faithful mock ASR/LLM/TTS | `.github/workflows/ci.yml`, `app/mocks/{mock_asr,mock_llm,mock_tts}.py`, `Makefile` · `mock_*.serve()` | A PR runs CI to green; `make test-integration` passes on a laptop using mocks only **(done)** |
| **L4.2** | 2 | Real pipeline behind one entrypoint; first full Hindi round-trip | `app/main.py`, `app/session.py` · `async run_turn(ctx)` — ASR stream → `AgentGraph.run` → TTS stream, sharing ctx; barge-in sets `ctx.cancel` | One command boots everything; a scripted Hindi question completes end-to-end with a grounded spoken answer → `eval/results/l4_2_e2e_hi.wav` |
| **L4.3** | 3 | Automated eval harness → one report | `eval/harness.py`, `eval/metrics/{wer,groundedness,latency}.py`, `eval/data/` · `Harness.run_all()->Report`, `groundedness(answer,sources)->score` | `python -m eval.harness` produces `eval/results/report.md` with **WER, groundedness, routing accuracy, latency** — the single source of truth for every pitch number |
| **L4.4** | 4 | Demo UI (transcript + grounded answer w/ citations + interrupt) + live dashboard | `app/ui/demo.py`, `app/ui/dashboard.py` · `Dashboard.render(metrics_feed)` (fed by Lane 3's `MetricsFeed`) | Dry run with a teammate: **all four use cases** demoed through the UI; dashboard updates per turn; screenshots saved |
| **L4.5** | 5 | Fallback video + dress rehearsal + deterministic offline path | `demo/script_hi.yaml`, `demo/fallback.mp4` | Full dress rehearsal **Jul 23**; fallback video + offline player verified to run with **no network**; go/no-go checklist signed |

## What you depend on (integrate at the seams)
- Lane 1's real `StreamingASR` / `StreamingTTS` (drop in for the mocks).
- Lane 2's `AgentGraph.run` (replaces the `_agent_glue` placeholder in `app/main.py`).
- Lane 3's `MetricsFeed` (`kisan_sarthi/serving/metrics.py`) for the dashboard, and `bench.py`
  outputs for the baseline-vs-current latency story.

> The mocks mean you can build and test the wiring, the harness scaffolding, and the dashboard
> shell **before any lane is real** — then swap each mock for the real component as it lands.