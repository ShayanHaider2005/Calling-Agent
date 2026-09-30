# LATENCY_REPORT — honest measured numbers

Measured on this machine: NVIDIA MX330 (2 GB VRAM), CPU-only torch (no CUDA),
7.8 GB RAM, Python 3.12. Date: 2026-09-30. Raw data: `results/benchmark_results.json`,
`results/tts_check.json`.

## Per-stage latency

| Stage | Latency | Notes |
|---|---|---|
| VAD | ~1 ms | energy-based, 16 kHz |
| Brain | 0.1 ms mean, 2.3 ms max | rule-based, no LLM |
| TTS (Piper) | 1.3 s after warmup (4-6 s first call) | per sentence, CPU |
| STT (whisper-tiny, CPU) | **40-54 s** | see caveat below |
| **End-to-end (full pipeline)** | **42-62 s** | dominated by STT |

## Resources

- Peak VRAM: **0 MB** (CPU-only torch; no CUDA build installed to stay under the
  6 GB download budget). The MX330 is unused.
- RAM: 19 MB (brain only); **601 MB** with STT + TTS + brain loaded.

## The STT caveat (important)

The 40-54 s STT number was measured on a **synthetic tone**, not real speech.
whisper-tiny on CPU enters a runaway loop on non-speech tones and generates hundreds
of "." tokens, which is what makes it slow. Real speech produces few tokens and would
be faster — but on CPU, whisper-tiny still runs at roughly 5-10x realtime, so a
3 s clip would take ~15-30 s. **Either way, CPU STT does not meet the ~1.5 s target.**

## Target vs actual

- Target: end-to-end response delay under ~1.5 s.
- Actual: 42-62 s (pathological tone) / realistically ~15-30 s for real speech on CPU.
- **Not met on CPU.** The bottleneck is STT.

## Suggestions to reach the target

1. **STT on GPU** (the original plan): a CUDA torch build + whisper-small/tiny on the
   MX320 would cut STT to well under 1 s. Not done tonight because the CUDA wheels +
   nvidia libs are ~4 GB (download budget) and the MX330 has only 2 GB VRAM.
2. **Smaller/faster STT**: a distilled or quantized model, or a streaming STT that
   starts emitting partial results.
3. **Speculative/cached responses**: for very common queries (hours, price), detect
   intent from partial STT and start TTS early.
4. **Reduce TTS first-call cost**: pre-warm voices at startup (already ~1.3 s warm).

## Honest summary

The pipeline works end to end and the brain is effectively free (0.1 ms). TTS is
acceptable (~1.3 s warm). **STT on CPU is the single bottleneck** and is the first
thing to fix (GPU STT) before the ~1.5 s target is reachable.
