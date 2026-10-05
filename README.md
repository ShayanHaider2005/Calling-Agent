# Calling-Agent

I'm building a multilingual speech-to-speech phone-call agent for a business. It
listens to a caller, understands them, and replies out loud in the caller's own
language — following the caller when they switch languages mid-call. The priority
is Urdu and English (including natural Urdu-English mixed speech), then Hindi.

I chose a **chained pipeline** rather than one end-to-end model, because each
stage can be measured, improved and swapped independently:

```mermaid
flowchart LR
    A[Caller speech] --> B[EARS<br/>Whisper STT<br/>+ LoRA adapter]
    B --> C[Language detect]
    C --> D[BRAIN<br/>state machine<br/>+ small LM]
    D --> E[MOUTH<br/>Piper TTS]
    E --> F[Agent speech]
    B --> G[CALL HANDLING<br/>VAD, barge-in,<br/>language following, latency]
    G --> B
```

## Status: work in progress

This is a working prototype, not a product. The pipeline runs end to end and is
tested, but it is not yet fast or accurate enough for real calls. See
[Known limitations](#known-limitations) and the honest numbers below.

## Supported languages

| Language | Status | Notes |
|---|---|---|
| English | Working | STT and TTS both functional. Only language with a dev set so far. |
| Urdu | Working (TTS weak) | STT and TTS functional; TTS voice is weak (see TTS_NOTES.md). |
| Hindi | Skeleton | TTS voice is poor (DEMO-ONLY). No dev data yet. |

## GPU requirements

- **GPU:** NVIDIA MX330 (2 GB VRAM) — the STT model runs on GPU.
- **STT model:** OpenAI **Whisper-small** (244M params), LoRA fine-tuned in 8-bit.
  This is the largest Whisper variant that fits in a 1.6 GB VRAM cap on this machine.
- **Brain LLM + TTS:** CPU (Piper).
- A **low-memory mode** runs the STT on CPU when the GPU is busy or too full.

## Quick start

```bash
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\check_env.py
```

## Run the call UI (one command)

```bash
.venv\Scripts\python.exe scripts\run_demo.py
# open http://127.0.0.1:8000
```

A phone-call screen: Start/End Call, live waveform, who-is-speaking indicator,
live transcript, language badge, per-turn latency, mute, barge-in, scenario
selector (clinic/restaurant), language hint, text-input fallback, and a status
panel. See DEMO_GUIDE.md for a pre-demo checklist and 10 test calls.

## Run the agent (other modes)

```bash
.venv\Scripts\python.exe -m src.pipeline text      # text mode (no mic)
.venv\Scripts\python.exe -m src.pipeline file -i audio.wav
.venv\Scripts\python.exe -m src.pipeline mic      # live mic
```

## Record a test set (optional, for evaluation)

```bash
.venv\Scripts\python.exe scripts\record_testset.py
```

One sentence at a time; Enter to record, `r` to redo, `s` to skip. Saves 16 kHz
mono WAV to `data/testset_audio/` + a row in `data/metadata.csv`. This set is
**never used for training** — it only measures the model. See RECORDING_GUIDE.md.

## Run the baseline (STT WER)

```bash
.venv\Scripts\python.exe scripts\baseline.py --metadata data/devset_metadata.csv
```

## Train (overnight LoRA loop)

```bash
.venv\Scripts\python.exe engine.py smoke              # tiny run to prove the loop
.venv\Scripts\python.exe engine.py curriculum         # staged-difficulty training
.venv\Scripts\python.exe engine.py night --hours 4    # time-budgeted LoRA rounds
```

The loop trains in time-budgeted rounds, evaluates WER after each round, keeps
the best checkpoint, rolls back if worse, and logs to `results/train_log.csv`.
Re-run the same command to resume. Create a file named `STOP` in the project
root to stop gracefully.

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
.venv\Scripts\python.exe scripts\check_demo.py       # automated call-flow health check
```

## Results (only real measured numbers)

| Metric | Value | How measured |
|---|---|---|
| English STT WER (public dev set) | 0.0971 (best), 0.1015 (final) | `engine.py eval`, whisper-small, 20 clips |
| Brain latency | 0.1 ms mean | `benchmark.py`, 50 turns |
| TTS latency (warm) | 2.0–3.0 s | Piper, CPU |
| End-to-end latency (GPU, warm) | 7.7–9.7 s | `benchmark.py` |
| End-to-end latency (CPU) | 42–62 s | `benchmark.py` |
| Peak VRAM (training) | 381 MiB | whisper-small 8-bit + LoRA |
| Peak VRAM (inference) | 277 MB | whisper-small 8-bit |
| Simulated-call pass rate | 273/273 (100%) | `test_brain.py` bulk simulation |
| Unit tests | 43 passed | eval + brain + ingest + packs + augment |

Everything not listed here is **TBD** — I have not measured it, and I won't guess.

## Responsible use

- **The agent says it is an AI.** Its greeting states that it is an AI assistant
  and that the call may be recorded. It never claims to be human; if asked, it
  says it is an AI and offers a human handoff.
- **Call recording.** The greeting notifies the caller that the call may be
  recorded. Do not deploy this without complying with local recording laws.
- **Knowledge boundary.** The agent answers only from the business knowledge
  file (`scenarios/*.yaml`). It never invents facts, prices or promises; if
  unsure, it offers a human handoff.
- **Licenses.** Every model, voice and dataset is recorded in LICENSES.md.
  Components marked DEMO-ONLY (e.g. the Hindi TTS voice) must not be used in a
  commercial product without replacement.

## Known limitations

- **Latency is 7.7–9.7 s** (GPU), not the ~1.5 s target. STT (~5.7 s) is the
  bottleneck. The first call is slower (~47 s) while the model loads.
- **Urdu and Hindi TTS voices are weak.** Hindi is DEMO-ONLY until a better
  voice is found.
- **No user test set yet.** The only labelled data is a small public English dev
  set (a stand-in, clearly labelled — not my test set). Urdu/Hindi WER is unmeasured.
- **Synthetic training data.** Accuracy is limited by the small synthetic set.
- **No real phone lines.** The agent runs over mic/file/web, not a phone network.
- **Small-model ceiling.** Whisper-small on a 2 GB GPU has a lower accuracy
  ceiling than larger models on bigger hardware.

## Roadmap

1. **Real test set** — record Urdu/English/Hindi clips, get per-language WER.
2. **Better TTS** — find stronger Urdu/Hindi voices, run native-speaker tests.
3. **Lower latency** — pre-warm the model, use a faster/streaming STT.
4. **Language packs** — train per-language LoRA adapters, wire the release gate.
5. **Phone integration** — connect a real phone line (future phase).

## Layout

```
configs/train.yaml      training config
scenarios/              business knowledge (clinic.yaml, restaurant.yaml)
src/                    eval, brain, pipeline, tts, packs, augment
engine.py               overnight LoRA training loop
scripts/                run_demo, check_demo, baseline, record_testset, benchmark,
                        tts_check, web_demo, ingest_links, release_gate, extract_devset
packs/                  per-language packs (english, urdu, hindi)
data/                   dev set + prompts (audio git-ignored)
results/                eval/benchmark outputs (regenerable, git-ignored)
```

## License

See [LICENSES.md](LICENSES.md) for every model, voice and dataset and its license.
