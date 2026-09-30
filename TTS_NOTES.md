# TTS_NOTES — Mouth (text-to-speech) quality

Engine: **Piper** (local, CPU, MIT license) via `piper-tts`. Fallback: pyttsx3/SAPI5.
Voices: rhasspy/piper-voices (permissive licenses — see LICENSES.md).

## Intelligibility check (round-trip WER)

Method: synthesize a sentence with Piper TTS -> transcribe it back with Whisper STT
(whisper-tiny) -> compute WER against the source text. This measures whether the
synthesized speech is intelligible enough for the STT to understand it.
Caveat: this conflates TTS quality with the STT's ability to handle synthetic speech,
but it is a useful relative proxy. Full results in `results/tts_check.json`.

| Language | Voice | Round-trip WER | Verdict |
|---|---|---|---|
| English | en_US-amy-medium | 0.20 | Acceptable (the 0.20 is mostly "9 AM" vs "9am" casing, not intelligibility) |
| Spanish | es_ES-davefx-medium | 0.48 | Understandable but weak; needs native-speaker listening test |
| Urdu | ur_PK-fasih-medium | 0.78 | Weak; STT transcribes it as garbled Urdu |
| Hindi | hi_IN-rohan-medium | 1.00 | Poor; STT hears romanized garbage, not Devanagari speech |

## Synthesis speed (CPU)

- First utterance per language: ~4-6 s (voice loads).
- Subsequent utterances: ~0.5-0.9 s per sentence.
- Acceptable for a demo; the first-call voice load should be pre-warmed in production.

## What needs native-speaker listening tests

- **Urdu**: is the "fasih" voice actually intelligible to a native Urdu speaker, or is
  the high WER a whisper-tiny limitation? Try the other Urdu voice (aegis_female) and
  listen to `results/tts_check/urdu/`.
- **Hindi**: the "rohan" voice scores WER 1.0 — the STT hears English-like sounds.
  Try pratham/priyamvada voices and listen. If all Hindi voices are weak, Hindi TTS is
  **DEMO-ONLY** until a better voice is found.
- **Spanish**: "davefx" is a male voice; check if it sounds natural.

## Fallback

pyttsx3/SAPI5 works for all languages but uses system voices (quality varies, and
non-English voices may not be installed). It is the fallback if Piper fails.

## Recommendation

- English: good to use.
- Spanish: usable for demo, verify with a native speaker.
- Urdu/Hindi: weak — flag as needing better voices before any real use. Do not mark
  commercially usable until a native speaker confirms intelligibility.
