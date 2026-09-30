# LICENSES — models, voices, datasets

Every model/voice/dataset used, its license, and whether commercial use is allowed.
Prefer commercially usable licenses. Non-commercial-only components are marked **DEMO-ONLY**.

## STT (EARS)

| Component | License | Commercial use | Notes |
|---|---|---|---|
| openai/whisper-tiny | MIT | Yes | Off-the-shelf STT + LoRA fine-tuning base. |
| openai/whisper-small | MIT | Yes | Surveyed as an alternative; not used (RAM/CPU). |

## LLM (BRAIN)

| Component | License | Commercial use | Notes |
|---|---|---|---|
| (none by default) | — | — | Rule-based fallback is the default and is fully tested. |
| Qwen/Qwen2.5-1.5B-Instruct (optional) | Apache-2.0 | Yes | Optional LLM for intent/rephrase; off by default (RAM). |

## TTS (MOUTH)

| Component | License | Commercial use | Notes |
|---|---|---|---|
| piper-tts (Rhasspy Piper) | MIT | Yes | Local CPU TTS engine. |
| piper voices (rhasspy/piper-voices) | MIT / CC0 (per voice) | Yes | en_US-amy, hi_IN-rohan, ur_PK-fasih. Verify per-voice. |
| pyttsx3 (SAPI5 system voices) | Depends on installed OS voices | Check per voice | Fallback only. |
| Coqui TTS | MPL-2.0 (some models NC) | Mixed | **NOT USED** — no Windows wheels; surveyed only. |

## Datasets

| Component | License | Commercial use | Notes |
|---|---|---|---|
| hf-internal-testing/librispeech_asr_dummy | MIT (test fixture) | Yes | Surveyed for smoke test; blocked by ffmpeg + broken file paths. |
| synthetic (piper TTS, known transcripts) | MIT (piper) | Yes | Used for the smoke test; self-contained. |
| data/testset_audio (user recordings) | User-owned | User's | Never used for training; evaluation only. |

## TTS quality flags (see TTS_NOTES.md)

- English: acceptable.
- Spanish: usable for demo; verify with a native speaker.
- Urdu: **weak** (round-trip WER 0.78) — needs a better voice + native-speaker test before real use.
- Hindi: **weak / DEMO-ONLY** (round-trip WER 1.00) — needs a better voice before any use.

## Notes

- All components must be re-verified at integration time; this file is the source of truth.
- Anything marked DEMO-ONLY must not be used in a commercial product without replacement.
