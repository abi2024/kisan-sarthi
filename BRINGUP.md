# Box Bring-Up Runbook — C2 → C4 → C5

The three remaining **foundational** (Gate-0) milestones are all GPU-gated and all owned by
Lane 3. This runbook turns them into a same-day checklist for the moment the RunPod H100 is
live. Work top-to-bottom; each phase has an explicit **validation gate** that must go green
before the next.

The map is blunt about why C5 matters: *"If C5 isn't captured in Week 1, you have no
before/after story and the optimization work is unprovable. Capture it first."* C2 and C4 exist
to unblock C5.

The key architectural fact that makes this clean: the NVIDIA Voice Agent Blueprint's LLM is
**pluggable by environment variable** — its `NVIDIA_LLM_URL` points at any OpenAI-compatible
`/v1` endpoint. That endpoint is *our* `serving/server.py`. So C4 (the blueprint loop) and C5
(our `bench.py`) both talk to the same `http://<box>:8000/v1` we already built and tested.

---

## Phase C2 — provision + validate the box   (owner: Lane 3)

**Goal:** a reachable H100 with NGC/NIM access, a network volume for weights, and our GPU deps
installed.

1. **Provision the RunPod H100.** Note the box's public IP/SSH. (Our serving lane needs 1 H100;
   the *full* blueprint loop wants ≥ 2 GPUs — one for ASR+TTS NIMs, one+ for the LLM. Size
   accordingly, or run ASR/TTS on NVIDIA-hosted NIM endpoints first and only the LLM locally.)

2. **Attach a network volume** for weights and datasets and mount it (e.g. `/workspace/vol`).
   Do **not** bake multi-GB weights into container images — slow pulls on every restart. Keep
   `acts_sweep`-style large artifacts and model checkpoints on the volume only.

3. **NGC / build.nvidia.com access.** Create an NGC API key and export it on the box:
   ```bash
   export NGC_API_KEY=<your key>
   export NVIDIA_API_KEY=$NGC_API_KEY      # some examples read this name
   echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
   ```

4. **Clone our repo + install GPU deps:**
   ```bash
   git clone <our repo> && cd kisan-sarthi
   uv sync --extra gpu          # adds vllm + nvidia-modelopt on top of base deps
   ```

5. **Pick the first model id** (still open — lock with the team). Then:
   ```bash
   export KISAN_MODEL_ID=<nvidia/Nemotron-3-Nano checkpoint, or a Llama-3.x NIM to unblock>
   ```
   Risk-register fallback: if Nemotron 3 Nano is awkward on the box, start with a Llama-3.x NIM
   behind the identical contract and swap later — it's a config change.

### ✅ C2 validation gate
```bash
make box-check            # nvidia-smi shows the GPU(s); required env vars present
make serve &              # launches vLLM on :8000 (real model) — see serving/server.py
make check-serving        # GET /health -> 200 + non-empty completion (our C2 health check)
```
`nvidia-smi` green + `check-serving` PASS == C2 done. (`make serve --print-cmd` first if you want
to eyeball the exact `vllm serve` line before committing the GPU.)

---

## Phase C4 — the English blueprint loop   (owner: Lane 1 + Lane 3)

**Goal:** reproduce the upstream NVIDIA cascaded ASR→LLM→TTS English voice loop on the box,
unmodified, and record a `.wav`. This is the real reference the rest of Lane 1 builds from.

1. **Clone the blueprint.** Official reference:
   `https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent` (end-to-end Nemotron voice
   agent). For a single-GPU local dev variant, the Daily/Pipecat `nemotron-january-2026`
   reference and the NVIDIA `voice-agent-examples` repo are lighter starting points. Follow
   *their* README for NIM bring-up (Nemotron Speech ASR / Magpie TTS) — those steps are
   upstream-owned and version-specific; don't hand-roll them.

2. **Bring up ASR + TTS** (Nemotron Speech ASR, Magpie TTS NIMs), per the blueprint README.
   - Magpie TTS is a **non-commercial preview**. For any commercial framing, swap to the Riva /
     enterprise TTS NIM (risk register) — the blueprint supports this via its TTS env vars.

3. **Point the blueprint's LLM at our server** (this is the integration seam):
   ```bash
   export NVIDIA_LLM_URL=http://<box-ip>:8000/v1     # our vLLM endpoint from Phase C2
   export NVIDIA_LLM_MODEL=$KISAN_MODEL_ID
   ```
   To de-risk, you can first run the loop fully on NVIDIA-hosted endpoints to confirm it works,
   then flip `NVIDIA_LLM_URL` to ours and confirm again. If the loop behaves identically, the
   contract is proven end-to-end.

4. **Run one English round-trip** through the browser/WebRTC UI and **save the audio**:
   ```bash
   # however the blueprint exposes capture, write the result here:
   #   eval/results/c4_baseline.wav
   ```

### ✅ C4 validation gate
A live English voice round-trip completes, and `eval/results/c4_baseline.wav` exists and plays
back a coherent answer. (`.wav` files are committed — `.gitignore` keeps them on purpose.)

---

## Phase C5 — the naive latency baseline   (owner: Lane 3) — **the #1 Week-1 deliverable**

**Goal:** measure the un-optimized FP16 serving numbers on the real model and commit them. Every
later optimization (FP8 in Wk2, NVFP4 in Wk3) is reported as a delta vs this file.

With `make serve` running the real model on `:8000` from Phase C2:
```bash
make capture-baseline        # runs bench.py against the real endpoint, writes the JSON
```
This produces `eval/results/baseline.json` (the canonical naive baseline) and a labelled
`eval/results/l3_1_fp16.json` copy (the name L3.1's validation expects). The GPU sampler now runs
*during* the bench, so `gpu` reports `util_mean_pct` / `util_peak_pct` / `mem_used_peak_mb` under
load — not a post-run snapshot.

Notes for a meaningful baseline (not just a smoke test):
- Single-stream (`--concurrency 1`) is the canonical "naive turn latency". `capture-baseline`
  also runs a 1/4/8 concurrency sweep so L3.3's batching work has a curve to beat.
- Read **p50/p95 latency, TTFT, and aggregate tokens/sec** — not `mean_per_request` tok/s, which
  is a mean-of-ratios and overweights fast requests.
- `--n 50` so percentiles mean something.

### ✅ C5 validation gate
`eval/results/baseline.json` committed with real p50/p95 first-token + tokens/sec + GPU util.

```bash
git add eval/results/baseline.json eval/results/l3_1_fp16.json eval/results/c4_baseline.wav
git commit -m "C2-C5: real H100 baseline (FP16) + English blueprint round-trip"
```

---

## After this runbook: Gate 0 is fully closed

C1, C3, C6 are already green (laptop + CI). Completing C2–C5 above means **all of C1–C6 are
validated and lanes can split into their Week-2 work.** Lane 3's Week-2 begins immediately:
L3.2 (FP8 on TensorRT-LLM) — the first delta vs the baseline you just captured.

### Reference links
- Blueprint: https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent
- Blueprint landing / benchmarks: https://build.nvidia.com/nvidia/nemotron-voice-agent
- NVIDIA Pipecat examples: https://github.com/NVIDIA/voice-agent-examples
- Single-GPU local reference + write-up: https://www.daily.co/blog/building-voice-agents-with-nvidia-open-models/
