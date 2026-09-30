# Calling-Agent

A multilingual (English / Urdu / Hindi / Spanish) speech-to-speech phone-call agent.
It listens to a caller, understands them, and replies out loud in the caller's
language — following the caller if they switch languages mid-call.

Design: a **chained pipeline**, not one end-to-end model:

```
EARS (Whisper STT) -> BRAIN (state machine + small LLM) -> MOUTH (Piper TTS) -> CALL HANDLING (VAD, barge-in, language following, latency)
```

Priority: Urdu + English, including natural Urdu-English mixed speech.

## Status (overnight build)

Working and tested: skeleton, eval tooling, test-set prompts, brain + guardrails
(56/56 simulated calls), end-to-end pipeline (text/file/mic), TTS (4 languages),
latency benchmark, overnight LoRA training loop (smoke + resume + STOP proven),
local web demo, data intake dry-run.

**Not done / known gaps**: CPU STT is slow (15-50 s; the ~1.5 s target needs GPU STT),
Urdu/Hindi TTS voices are weak, test-set prompts are UNVERIFIED, no real phone lines.

See MORNING_REPORT.md for the full honest status.

## Install

```bash
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\check_env.py
```

## 1. Record the test set (optional, for evaluation)

```bash
.venv\Scripts\python.exe scripts\record_testset.py            # all prompts
.venv\Scripts\python.exe scripts\record_testset.py --lang urdu
```

One sentence at a time; press Enter to start/stop recording, `r` to redo, `s` to
skip. Saves 16 kHz mono WAV to `data/testset_audio/` + a row in `data/metadata.csv`.
See RECORDING_GUIDE.md. **This set is never used for training.**

## 2. Run the baseline (STT WER on the test set)

```bash
.venv\Scripts\python.exe scripts\baseline.py                    # whisper-small
.venv\Scripts\python.exe scripts\baseline.py --model openai/whisper-tiny --limit 10
```

Writes `results/baseline_results.json` (per-language + mixed/pure WER).

## 3. Run the agent

Text mode (no mic needed, fully testable):

```bash
.venv\Scripts\python.exe -m src.pipeline text
```

File mode (audio in, audio out):

```bash
.venv\Scripts\python.exe -m src.pipeline file -i path\to\audio.wav
```

Mic mode (live, with barge-in):

```bash
.venv\Scripts\python.exe -m src.pipeline mic
```

Web demo (localhost only):

```bash
.venv\Scripts\python.exe scripts\web_demo.py
# open http://127.0.0.1:8000
```

## 4. Run the overnight training loop

```bash
.venv\Scripts\python.exe engine.py smoke              # tiny run to prove the loop
.venv\Scripts\python.exe engine.py night --hours 4    # time-budgeted LoRA rounds
```

- Trains in time-budgeted rounds, evaluates WER after each round, keeps the best
  checkpoint, rolls back if worse, logs to `results/train_log.csv`.
- **Resume**: just run the same command again — it continues from the last round.
- **STOP**: create a file named `STOP` in the project root to stop gracefully.
- Checkpoints go to `checkpoints/` (git-ignored); best to `checkpoints/best/`.

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

## Layout

```
configs/train.yaml     training config
scenarios/clinic.yaml   fictional business knowledge (the brain answers only from this)
src/eval.py             WER evaluation (per-language, Urdu/Hindi normalization)
src/brain.py            state machine + intent + guardrails
src/pipeline.py         end-to-end pipeline (VAD/STT/brain/TTS, latency log)
src/tts.py              Piper TTS + SAPI5 fallback
scripts/                baseline, record_testset, benchmark, tts_check, web_demo, ingest_links
engine.py               overnight LoRA training loop
data/testset_prompts/   150 prompts (UNVERIFIED)
data/metadata.csv       test-set metadata (created by record_testset.py)
results/                eval/benchmark/tts outputs (regenerable)
```

## License

See LICENSES.md for every model/voice/dataset and its license. Components marked
DEMO-ONLY (e.g. Hindi TTS) must not be used commercially without replacement.
