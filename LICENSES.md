# LICENSES — models, voices, datasets

Every model/voice/dataset used, its license, and whether commercial use is allowed.
Prefer commercially usable licenses. Non-commercial-only components are marked **DEMO-ONLY**.

## STT (EARS)

| Component | License | Commercial use | Notes |
|---|---|---|---|
| openai/whisper-small (off-the-shelf) | MIT | Yes | Baseline + fine-tuning base. |
| openai/whisper-tiny | MIT | Yes | Fallback / smoke tests. |

## LLM (BRAIN)

| Component | License | Commercial use | Notes |
|---|---|---|---|
| (TBD — small 1.5–3B model, 4-bit) | | | |

## TTS (MOUTH)

| Component | License | Commercial use | Notes |
|---|---|---|---|
| piper-tts (Rhasspy Piper) | MIT | Yes | Local CPU TTS, Windows pip wheel. |
| piper voices (rhasspy/piper-voices) | Mostly MIT / CC0 | Yes | Per-voice license files ship with voices; verify per voice. |
| pyttsx3 (SAPI5 system voices) | Depends on installed OS voices | Check per voice | Fallback only; quality lower. |
| Coqui TTS | MPL-2.0 (some models NC) | Mixed | **NOT USED** — no Windows wheels; noted as surveyed. |

## Datasets

| Component | License | Commercial use | Notes |
|---|---|---|---|
| (TBD — smoke-test dataset) | | | |

## Notes

- All components must be re-verified at integration time; this file is the source of truth.
- Anything marked DEMO-ONLY must not be used in a commercial product without replacement.
