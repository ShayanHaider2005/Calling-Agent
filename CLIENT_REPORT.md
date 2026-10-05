# Client Report — Calling-Agent (confidential demo)

**Audience:** a business owner evaluating a phone-call AI agent.
**Status:** working prototype, **not production-ready**. All numbers below are
measured on the test machine, not estimates.

## What the agent does (in plain words)

The agent answers a phone call, listens to the caller, understands what they want,
and speaks back in the caller's own language. It can:

- Answer questions about opening hours, prices, services and location.
- Take an appointment booking (name, day, time) and confirm it.
- Switch languages mid-call when the caller does (Urdu <-> English, Hindi <-> English).
- Say it is an AI and that the call may be recorded (it never claims to be human).
- Offer to connect the caller to a human when it doesn't know something.

It is a **chained pipeline**: speech-to-text (Whisper) -> a rule-based brain ->
text-to-speech (Piper). There is no single "AI brain"; each part is small and
measurable.

## Demo scenario (clearly fictional)

The demo uses **"Sunrise Family Clinic"**, a fictional business with invented
hours, prices and address. A second scenario (**"Bistro Demo"**, a restaurant) is
included to show the business data is swappable. **None of these businesses are real.**

## Languages supported (honest status)

| Language | Status |
|---|---|
| English | Working. Only language with a dev set so far. |
| Urdu | Working. TTS voice is weak — needs a better voice before real use. |
| Hindi | Skeleton. TTS voice is poor (DEMO-ONLY). No dev data yet. |

Urdu-English mixed speech is the priority and is handled. Hindi-English mixing
is second.

## STT model and hardware

- **Model:** OpenAI **Whisper-small** (244M parameters), fine-tuned with LoRA
  adapters (2.6% of params trained) in 8-bit quantization.
- **Hardware:** NVIDIA MX330 GPU (2 GB VRAM), 7.8 GB RAM, Windows.
- **Why this size:** it is the largest Whisper variant that trains and runs
  within a 1.6 GB VRAM cap on this machine. A smaller model means a lower
  accuracy ceiling — this is an honest limitation.

## Measured results (before vs after, same test set)

Test set: 20 English clips from a public dev set (librispeech_dummy). WER = word
error rate (lower is better). Measured with `scripts/baseline.py` and `engine.py eval`.

| Metric | Before (base Whisper-small) | After (LoRA fine-tuned) |
|---|---|---|
| English WER (dev set) | 0.1104 | 0.0971 (best), 0.1015 (final eval) |
| Simulated-call pass rate | — | 273/273 (100%) |

The improvement is real but modest. The training data is synthetic (piper-TTS
clips with known transcripts) plus augmentation (noise, 8 kHz phone quality).
**Urdu and Hindi WER are not yet measured** — there is no dev data for them.

## Response-time numbers (measured)

| Mode | End-to-end latency |
|---|---|
| CPU | 42–62 s |
| GPU (warm) | 7.7–9.7 s |
| GPU (first call, cold) | 46.9 s (model + voice loading) |

The ~1.5 s target is **not met**. The speech-to-text step (~5.7 s on GPU) is the
bottleneck. Pre-warming the model at startup would make the first call fast.

## Example conversations (automated tests, not real calls)

These are from `scripts/check_demo.py`, which plays synthetic caller audio into
the pipeline and checks the reply. **They are automated tests, not real callers.**

**Call 1 (English):**
- Caller: "Hello, I want to book an appointment."
- Agent: "Hello, thank you for calling Sunrise Family Clinic. This is an AI
  assistant, and this call may be recorded. How can I help you today?"

**Call 2 (English):**
- Caller: "What are your opening hours?"
- Agent: "We are open Monday to Friday from 9 AM to 5 PM, and on Saturday from
  10 AM to 2 PM. We are closed on Sundays and public holidays."

## Current limitations (stated plainly)

1. **Not production-ready.** It is a prototype for evaluation, not a product.
2. **Urdu/Hindi TTS is weak.** Hindi is DEMO-ONLY; Urdu needs a better voice.
3. **Latency is 7.7–9.7 s** (GPU), not the ~1.5 s target.
4. **No real test set.** Only a public English dev stand-in exists.
5. **Synthetic training data.** Real per-language data would improve accuracy.
6. **No real phone lines.** The agent runs over mic/file/web, not a phone network.
7. **Small-model ceiling.** Whisper-small on a 2 GB GPU has a lower accuracy
   ceiling than larger models on bigger hardware.

## Data, privacy and call recording

- The agent **says it is an AI** and that **the call may be recorded** (in the greeting).
- It **never claims to be human**; if asked, it says it is an AI and offers a human handoff.
- It **answers only from the business knowledge file** — it never invents facts,
  prices or promises.
- Call recordings (if enabled) are stored locally and are **never uploaded**.
  They are git-ignored and contain no personal data by default.
- **Do not deploy without complying with local call-recording laws.**

## Licensing status

| Component | License | Commercial use |
|---|---|---|
| Whisper-small | MIT | Yes |
| Piper TTS + voices | MIT / CC0 | Yes |
| Qwen2.5-1.5B (optional LLM) | Apache-2.0 | Yes |
| **Hindi TTS voice** | — | **DEMO-ONLY — replace before real use** |
| Public dev set | MIT (test fixture) | Evaluation only |

## Proposed next steps for a small pilot

1. **Record a real test set** (Urdu/English/Hindi) to measure per-language accuracy.
2. **Find better Urdu/Hindi TTS voices** and run native-speaker listening tests.
3. **Reduce latency** — pre-warm the model, use a faster/streaming STT.
4. **Collect real training data** (permissively licensed) to improve accuracy.
5. **Run a small pilot** with a handful of real calls and human supervision.

---

*This report contains only measured facts. Anything not listed here has not been
measured and should not be assumed to work.*
