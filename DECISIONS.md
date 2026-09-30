# DECISIONS — choices made and why

## Hardware reality (measured, not assumed)

- GPU: NVIDIA **MX330, 2048 MiB VRAM** (the task assumed 4 GB). Driver 581.42, CUDA 13.0.
- RAM: **7.8 GB total, ~1.6 GB free** at start.
- Python 3.12.10 via the `py` launcher (`python` is the Microsoft Store stub).

## Key decisions

1. **CPU-only torch** (`torch 2.14.0+cpu`). The CUDA wheels + nvidia libs are ~4 GB,
   which would blow the 6 GB download budget. Consequence: STT runs on CPU and is the
   latency bottleneck (see LATENCY_REPORT.md). The MX330 is unused. This is the first
   thing to revisit (GPU STT) to reach the ~1.5 s target.

2. **whisper-tiny** (not whisper-small) for STT. Smaller footprint, fits in RAM, and
   is the practical choice on this machine. whisper-small is a config option.

3. **Rule-based brain by default (no LLM).** With ~1.6 GB free RAM, a 1.5B 4-bit LLM
   is risky. The rule-based fallback is fully tested (56/56 simulated calls). The LLM
   (Qwen2.5-1.5B-Instruct, Apache-2.0) is wired in as an optional classifier/rephraser
   behind `--use-llm`, off by default.

4. **Piper for TTS** (not Coqui TTS). Coqui TTS has no Windows wheels. Piper is MIT,
   local, CPU, and has voices for all 4 languages. pyttsx3/SAPI5 is the fallback.

5. **Synthetic data for the training smoke test.** The public dummy dataset
   (librispeech_dummy) is blocked by ffmpeg (audio decode) and broken file paths.
   Synthetic piper-TTS clips with known transcripts are self-contained and prove the
   loop. Real training should use `data/train/` (local) or a curated permissive dataset.

6. **Energy-based VAD** (not webrtcvad). Dependency-light, works on any platform, and
   is unit-tested. webrtcvad is finicky on Windows.

7. **Audio passed as numpy arrays** to the STT pipeline (not file paths), because the
   transformers ASR pipeline uses ffmpeg to read files and ffmpeg is not installed
   (and system installs are forbidden). soundfile reads the WAV/PCM directly.

8. **Language following is per-utterance.** The agent switches language on every
   utterance the caller speaks (script-decisive for Urdu/Hindi, stopword-based for
   English/Spanish), so it follows switches in both directions.

## Trade-offs / honest notes

- The ~1.5 s latency target is **not met on CPU** (STT is 15-50 s). Brain is 0.1 ms,
  TTS ~1.3 s warm. GPU STT is the fix.
- Urdu and Hindi TTS are weak (round-trip WER 0.78 / 1.00). Flagged in TTS_NOTES.md;
  Hindi is DEMO-ONLY until a better voice is found.
- The test-set prompts are UNVERIFIED until the user checks them.
