# Lane 2 — Agent & Data / RAG · Deliverables

**Owner:** Agent engineer · **Owns:** `kisan_sarthi/agent/` and `kisan_sarthi/data/`
**Mandate:** the brain and its food — the multi-agent workflow (intent → RAG → tools → guardrail
→ privacy router), the knowledge base (scheme rules, KCC corpus, guideline PDFs), the live tools
(mandi prices, weather), the synthetic data, and the responsible-AI guardrails. Develops with a
text-in/text-out harness against the mock ASR/TTS.

**Contracts you produce/consume** (frozen — see `kisan_sarthi/contracts/models.py`): you consume
`ASRResult` and stream `AgentEvent`s (text deltas → tool → final/escalate, each carrying
`Source` citations). You call the LLM through **`kisan_sarthi/serving/client.py`** (the stock
`openai` SDK against the `/v1` endpoint) — **the LLM seam is the OpenAI HTTP API, not a
dataclass.** Your `AgentGraph.run` replaces the `_agent_glue` placeholder in `app/main.py`.

> You can build the entire agent **today, zero GPU**, against the mock LLM
> (`uv run uvicorn app.mocks.mock_llm:app --port 8000`). Caveat: the mock returns a *fixed canned
> reply* regardless of input — perfect for wiring streaming, tool calls, citation structure and
> guardrail flow, but real answer quality needs the model on the box.

**Definition of success:** given a transcribed question, the agent identifies intent, retrieves
grounded facts, calls live tools when needed, refuses/escalates anything it can't ground, **never
gives a binding financial/eligibility verdict**, and streams `AgentEvent`s — all within ~50–150 ms
orchestration budget (excluding the LLM call).

## Milestones

| ID | Wk | Deliverable | Key files / signatures | ✅ Validation gate (what "done" means) |
|----|----|-------------|------------------------|----------------------------------------|
| **L2.1** | 1 | KB ingestion + RAG retriever returning cited passages | `data/ingest/build_kb.py`, `data/sources/`, `agent/rag/retriever.py` · `build_kb(src,out)`, `Retriever.search(q,k)->list[Passage]`, `Passage.to_citation()->Source` | `python -m agent.rag.smoke "PMFBY claim window"` cites the guidelines with the 72-hour figure; **recall@5 ≥ 0.8** on `eval/data/rag_questions.jsonl` |
| **L2.2** | 1–2 | Machine-readable eligibility rules (the data that doesn't exist structured anywhere) | `data/rules/schemes.json`, `agent/eligibility.py` · `EligibilityEngine.check(profile,scheme)->Indicative` (may_qualify + reasons + docs, **never** binding) | Unit tests cover **8+ priority schemes** (PM-KISAN, PMFBY, KCC, PM-KMY, NSAP, PDS, MGNREGA + 1 state); `pytest test_eligibility.py` green; every output uses "confirm with…" language |
| **L2.3** | 2 | Agent graph: intent classify → route → tools (live mandi + weather) | `agent/graph.py`, `agent/intents.py`, `agent/tools/{mandi,weather,scheme_lookup}.py` · `async AgentGraph.run(ctx,asr)->AsyncIterator[AgentEvent]`, `classify_intent(text)->Intent`, `MandiTool.get_price(...)`, `WeatherTool.forecast(...)` | Text harness: "aaj Indore mandi mein soybean ka bhaav?" triggers MandiTool w/ grounded price; "gehu ke liye loan?" routes to credit; **routing accuracy ≥ 0.9** on 30-q set → `eval/results/l2_3_intents.json` |
| **L2.4** | 3 | Guardrails (NeMo) + grounding enforcement + privacy router + escalation | `agent/guardrails/config/`, `agent/guardrails/grounding.py`, `agent/privacy_router.py`, `agent/escalation.py` · `enforce_grounding(answer,passages)`, `PrivacyRouter.route(text)->'local'\|'cloud'`, `should_escalate(intent,conf)->bool` | Adversarial set `eval/data/guardrail_probes.jsonl`: **100% handled** — refuses binding loan approval, never an ungrounded number, escalates PII. This is the gate. |
| **L2.5** | 4 | Synthetic Hinglish dialogues (NeMo Data Designer) + optional LoRA | `data/synthetic/generate.py`, `data/synthetic/seeds/`, `agent/finetune/lora_config.yaml` · `generate_dialogues(n,seeds)->dataset` | **≥ 5K validated dialogues** in `data/synthetic/out/` passing schema + LLM-judge; if LoRA run, tone improvement vs base → `eval/results/l2_5_lora.json` |

## Targets
RAG recall@5 ≥ 0.8 · intent routing ≥ 0.9 · **100%** of guardrail probes handled · orchestration
latency ~50–150 ms (excl. LLM).

## Critical note
**L2.4 is the responsible-AI core — do not ship without it.** A confident wrong financial/eligibility
answer to a low-literacy user is the worst-case failure. Ground every claim, cite sources, refuse
or escalate otherwise, never a binding verdict.

**Scope guard:** L2.5 (synthetic + LoRA) is an *enhancement*. If Week 4 is tight, ship the
RAG-grounded agent without LoRA — do not let it block integration.