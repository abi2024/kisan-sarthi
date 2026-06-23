# Lane 3 — Serving & Optimization · Deliverables

**Owner:** Abishek (serving/perf) · **Owns:** `kisan_sarthi/serving/` and `eval/perf/`
**Mandate:** serve Nemotron 3 Nano fast, quantize it, profile the whole pipeline, and own the
latency/throughput scoreboard. This is the lane that wins on measurable performance — it starts
Week 1 (captures the baseline) and optimizes to the finals.

**Contracts you expose** (frozen — see `kisan_sarthi/contracts/models.py`): the LLM seam **is**
the OpenAI-compatible HTTP API (`POST /v1/chat/completions`, `stream=True`,
`extra_body={reasoning,max_tokens}`, cancel via client abort). No custom request dataclass. The
model is set by `KISAN_MODEL_ID` — model-agnostic, never hard-coded — so the whole lane's
fallbacks ("swap is a config change") hold.

**State now:** the scaffold is built and green on a laptop (vLLM launcher, OpenAI client, bench
harness, FP8/NVFP4 stubs, in-memory metrics). The GPU-gated foundational items (**C2 provision →
C4 blueprint loop → C5 real baseline**) have a same-day checklist in **[BRINGUP.md](../../BRINGUP.md)**.

**Definition of success:** the model is served behind the OpenAI seam, quantized FP8→NVFP4 with
quality retained, the full pipeline profiled, and the end-to-end p50 turn latency driven under
800 ms — every gain a measured delta vs the FP16 baseline.

## Milestones

| ID | Wk | Deliverable | Key files / signatures | ✅ Validation gate (what "done" means) |
|----|----|-------------|------------------------|----------------------------------------|
| **L3.1** | 1 | Serve Nemotron behind the OpenAI endpoint (vLLM first); capture FP16 baseline | `serving/server.py`, `serving/client.py`, `eval/perf/bench.py` · `serve(model,dtype,port)`, `bench.run(prompts,concurrency)->Metrics` | `python -m eval.perf.bench --dtype fp16` writes `eval/results/l3_1_fp16.json` with **p50/p95 first-token + tokens/sec** — the number every optimization is measured against (see `make capture-baseline`) |
| **L3.2** | 2 | TensorRT-LLM + FP8 (ModelOpt); verify quality, measure speedup | `serving/trtllm/build_engine.py`, `serving/quantize.py` · `build_engine(checkpoint,precision)`, `verify_quality(engine,eval_set)->score` (~99% retention) | `eval/results/l3_2_fp8.json`: **tokens/sec up vs FP16 at equal task quality**; speedup factor recorded for the deck |
| **L3.3** | 3 | NVFP4 (4-bit) + KV-cache/batching tuning | `serving/quantize.py` (NVFP4 path), `serving/config/serving.yaml` · `tune_kv_and_batch(grid)->best_config` | `eval/results/l3_3_nvfp4.json`: **~21 GB MoE footprint**, tokens/sec, quality retained; best serving config committed |
| **L3.4** | 4 | Nsight profiling + per-leg metrics feed for the dashboard | `eval/perf/profile_nsight.sh`, `serving/metrics.py` · `profile_turn()->trace`, `MetricsFeed.emit(leg,ms)` | Nsight trace + per-leg breakdown → `eval/results/l3_4_profile/`; **top bottleneck identified**; dashboard shows baseline-vs-current |
| **L3.5** | 5 | Final push to target (handoff overlap, spec-speech, length caps, reasoning off) | `serving/config/serving.yaml` (final), `eval/perf/bench.py` · `measure_e2e()->turn_latency` | `eval/results/l3_5_final.json`: **end-to-end p50 turn latency < 800 ms**; baseline→final improvement table finalized |

## Targets
FP16 baseline → FP8 → NVFP4 each measured (speedup + footprint + quality) · end-to-end p50 turn
latency **< 800 ms**.

## Fallbacks (risk register)
- TRT-LLM Nemotron immature on Spark/Blackwell → **vLLM FP8** (still a real speedup, cleaner). Pin `nvidia-modelopt ≥ 0.17`.
- Nemotron 3 Nano unavailable → a Llama-3.x NIM behind the identical endpoint; swap is one env var.
- Latency > 800 ms → reasoning OFF + capped budget, streaming TTS by sentence, 80–160 ms ASR chunks, NVFP4.

> Critical: **capture C5 first.** Without the baseline in Week 1, every optimization is unprovable.
> `make box-check` → `make serve` → `make capture-baseline` the day the H100 lands.