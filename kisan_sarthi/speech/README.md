# Lane 1 — Speech I/O · Deliverables

**Owner:** Speech engineer · **Owns:** `kisan_sarthi/speech/`
**Mandate:** everything between the microphone and the speaker — streaming ASR, streaming TTS,
VAD, barge-in, telephony transport, and Indic / code-switch speech quality. Develops in
isolation with pre-recorded clips; needs no agent or LLM to make progress.

**Contracts you produce/consume** (frozen — see `kisan_sarthi/contracts/models.py`): you emit
`ASRResult` (Lane 1 → 2) and `TTSChunk` (Lane 2 → 1), and you consume the agent's `AgentEvent`
stream for synthesis. Your mock counterparts already pin your signatures:
`app/mocks/mock_asr.py` and `app/mocks/mock_tts.py` — replace their bodies with real NIM-backed
implementations and the rest of the system keeps working unchanged.

**Definition of success:** a farmer speaks Hindi with English words mixed in, is transcribed
correctly (WER within target), can interrupt the agent mid-sentence and have it stop within
~300 ms, and hears a natural Indic voice whose first audio arrives ~250–500 ms after the LLM's
first token.

## Milestones

| ID | Wk | Deliverable | Key files / signatures | ✅ Validation gate (what "done" means) |
|----|----|-------------|------------------------|----------------------------------------|
| **L1.1** | 1 | Streaming ASR (English) behind `ASRResult`, interim+final over WS, turn_id-stamped | `speech/asr/streaming_asr.py`, `speech/asr/vad.py`, `speech/contracts_impl.py` · `async StreamingASR.stream(ctx)->AsyncIterator[ASRResult]`, `StreamingASR.push_audio(chunk)`, `VAD.is_speech(frame)` | `python -m speech.asr.smoke` on `eval/data/asr_smoke/*.wav` prints finals; **median ASR-final < 200 ms** → commit `eval/results/l1_1_asr_en.json` |
| **L1.2** | 1 | Streaming TTS (English) behind `TTSChunk`, synthesizing **per sentence** | `speech/tts/streaming_tts.py`, `speech/tts/sentence_splitter.py` · `async StreamingTTS.synthesize(events,ctx)->AsyncIterator[TTSChunk]`, `SentenceSplitter.feed(delta)->list[str]` | `python -m speech.tts.smoke "block my card please"` → **first-chunk < 400 ms**; save `eval/results/l1_2_tts_en.wav` + timing log |
| **L1.3** | 2 | Hindi + code-switch (Hinglish: Roman+Devanagari), domain word-boosting | `streaming_asr.py` (+lang/boost), `speech/asr/word_boost.py`, `streaming_tts.py` (Hindi speaker) · `StreamingASR.set_language(lang,boost_terms)`, `normalize_codeswitch(text)->str` | Hinglish clips (`eval/data/asr_codeswitch/`) → WER + code-switch WER to `eval/results/l1_3_wer.json` (tracked gate; clean Hindi WER competitive) |
| **L1.4** | 3 | Barge-in + end-of-utterance: stop TTS, flush, mark context interrupted | `speech/barge_in.py`, `speech/eou.py` · `BargeInController.on_user_speech()`, `EOU.should_finalize(vad_state,silence_ms)->bool` | Inject user audio at t=1s into a long reply; **TTS stops < 300 ms** → `eval/results/l1_4_bargein.json`; agent re-listens |
| **L1.5** | 5 | Pipecat+WebRTC phone surface + deterministic offline replay | `speech/transport/webrtc_pipe.py`, `speech/transport/offline_player.py` · `build_pipeline(agent,asr,tts)`, `OfflinePlayer.run(script_path)` | A real phone/browser call completes a full Hindi turn; `python -m speech.transport.offline_player demo/script_hi.yaml` runs with **no network** |

## Targets
ASR-final latency < 200 ms · TTS first-chunk < 400 ms · barge-in stop < 300 ms · code-switch
WER tracked (escalate word-boosting / fine-tune if > 25%).

## Fallbacks (risk register)
- ASR awkward on the box → Riva Parakeet streaming → faster-whisper chunked (higher latency, still demoable).
- Magpie TTS is **non-commercial preview** → enterprise/Riva TTS NIM for commercial framing; AI4Bharat IndicF5 as fallback.

> Build against mocks now: nothing here blocks on the GPU or other lanes. Your smoke tests run on
> committed `.wav` clips. Integrate only at the `ASRResult` / `TTSChunk` / `AgentEvent` seams.