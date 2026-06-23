# First GPU Session — Execution Plan

The H100 is metered. The goal of session one: walk in with everything decided and staged, and
walk out with **C2 + C4 + C5 green and committed** (and L3.2 started if time allows) — having
spent the GPU only on things that *need* a GPU. Target: ~2–3 focused hours.

The rule that saves the most money: **the meter pays for tokens/sec, not for thinking.** Every
decision, every command, every dataset is prepared on CPU *before* you start the box. While big
downloads/builds run, you do the next prep step — never sit watching a progress bar idle.

---

## Pre-flight — finish ALL of this on the laptop before you start the box

A session is "ready to start" only when every box below is checked:

- [ ] **Model id locked.** `KISAN_MODEL_ID` decided (Nemotron-3-Nano, or the Llama-3.x NIM
      fallback). You know the exact NGC/HF string. *(This is the one decision that, if unmade,
      wastes the whole session.)*
- [ ] **Tokens in hand.** NGC API key created; HF token if the checkpoint needs one. Pasted into
      a local scratch file ready to export — not hunted for on the box.
- [ ] **Repo pushed & green.** CI passing on `main`; the box will `git clone` a known-good tree.
- [ ] **Commands staged.** A scratch file with the exact command sequence below, model id filled
      in, so you paste, not type.
- [ ] **Eval inputs staged.** Any English ASR clips you want for C4/C5 sanity already in the repo
      (or a URL ready). `eval/data/samples/*` is enough to run the harness.
- [ ] **CPU work merged.** Session manager + eval harness are in (done). On the box you only add
      real numbers, not new code.
- [ ] **Reasoning-parser plugin URL ready.** Nemotron-3-Nano needs `nano_v3_reasoning_parser.py`
      fetched into the repo root before `make serve` (the exact `wget` is in Phase 1). Have the URL
      ready, or plan to comment `reasoning_parser_plugin` out of `serving.yaml` for the baseline.
- [ ] **Reasoning-parser plugin URL ready.** Nemotron-3-Nano needs `nano_v3_reasoning_parser.py`
      fetched into the repo root before `make serve` (the exact `wget` is in Phase 1). Have the URL
      ready, or plan to comment `reasoning_parser_plugin` out of `serving.yaml` for the baseline.
- [ ] **Blueprint README skimmed.** Read the NVIDIA nemotron-voice-agent README *now* so the NIM
      bring-up isn't a first-read on the meter. (See BRINGUP.md links.)

If any box is unchecked, you are not ready — finishing it on the box costs GPU-hours.

---

## Phase 0 · Boot & clone (~5 min, low GPU use)
```bash
nvidia-smi                                   # confirm the GPU(s) are visible
git clone https://github.com/abi2024/kisan-sarthi && cd kisan-sarthi
export KISAN_MODEL_ID=<locked checkpoint>
export NGC_API_KEY=<key>; export NVIDIA_API_KEY=$NGC_API_KEY
uv sync --extra gpu                          # <-- this pulls vLLM etc; KICK IT OFF, then keep reading
```
**While `uv sync --extra gpu` runs:** mount the network volume, `docker login nvcr.io`, and start
pulling the model weights / NIM containers in another shell. Overlap the downloads.

## Phase 1 · C2 — box ready (~5 min once downloads finish)
```bash
make box-check                               # nvidia-smi + env + vllm importable
# Nemotron-3-Nano needs its reasoning-parser plugin in the working dir (vLLM resolves it
# from CWD). Fetch it into the repo root before serving:
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py
#   (or comment out `reasoning_parser_plugin` in serving.yaml — the latency baseline doesn't
#    need reasoning-trace separation. `make serve` warns if the file is missing.)
# Nemotron-3-Nano needs its reasoning-parser plugin in the working dir (vLLM resolves it
# from CWD). Fetch it into the repo root before serving:
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py
#   (or comment out `reasoning_parser_plugin` in serving.yaml — the latency baseline doesn't
#    need reasoning-trace separation. `make serve` warns if the file is missing.)
make serve &                                 # vLLM serves $KISAN_MODEL_ID on :8000
#   (first run downloads/loads weights — this is the big wait; let it; do Phase 2 prep meanwhile)
make check-serving                           # GET /health 200 + non-empty completion  -> C2 PASS
```
**Gate:** `nvidia-smi` green + `check-serving` PASS. **C2 done.**
**While weights load:** in another shell, start the ASR/TTS NIMs for Phase 2 so they're warm.

## Phase 2 · C5 FIRST — capture the baseline (~10 min) ⚠️ do this before C4
```bash
make capture-baseline                        # bench vs the real model -> baseline.json + l3_1_fp16.json + sweep
```
**Why before C4:** C5 only needs the LLM endpoint (up in Phase 1) — it does **not** need the full
speech loop. The baseline is the #1 deliverable and the thing every optimization is measured
against, so bank it the moment the model serves, before spending time on the blueprint. If the
session gets cut short, you still leave with the baseline.
**Gate:** `eval/results/baseline.json` has real p50/p95 TTFT + tokens/sec + a non-null `gpu` block
(sampled under load). **C5 done.**

## Phase 3 · C4 — the English blueprint loop (~30–60 min, the long pole)
Bring up the NVIDIA voice-agent blueprint (ASR + TTS NIMs per its README), then point its LLM at
the server already running from Phase 1:
```bash
export NVIDIA_LLM_URL=http://<box-ip>:8000/v1
export NVIDIA_LLM_MODEL=$KISAN_MODEL_ID
# run one English voice round-trip through the blueprint UI; save the audio:
#   -> eval/results/c4_baseline.wav
```
**Gate:** a live English round-trip completes; `c4_baseline.wav` saved. **C4 done.**
De-risk: if the loop misbehaves, run it first against NVIDIA-hosted endpoints, confirm, then flip
`NVIDIA_LLM_URL` to ours.

## Phase 4 · Commit, then either push into L3.2 or tear down
```bash
git add eval/results/baseline.json eval/results/l3_1_fp16.json eval/results/c4_baseline.wav eval/results/report.md
git commit -m "C2-C5: real H100 FP16 baseline + English blueprint round-trip"
git push
```
**Now Gate 0 is fully closed.** If GPU time remains, start L3.2 (FP8): `serving/quantize.py` +
`serving/trtllm/build_engine.py`, then `bench --label fp8` and compare to baseline. If not, **stop
the box** — the artifacts are committed, so the next session starts from a known state.

---

## Stop-the-box checklist (every session)
- [ ] Baseline + any new result JSON committed and pushed (don't lose data to a killed pod).
- [ ] Weights/engines on the **network volume**, not the ephemeral container disk.
- [ ] `report.md` regenerated (`python -m eval.harness`) so the numbers are captured.
- [ ] Pod stopped/terminated. Confirm in the RunPod console — a forgotten running pod is pure burn.

## What NOT to do on the GPU (do these on CPU)
- Writing/debugging non-serving code, editing docs, fiddling with the agent or harness logic.
- Reading the blueprint README for the first time.
- Deciding the model id or hunting for API keys.
- Anything the mock already validated — the wiring is proven; the box is for real numbers only.