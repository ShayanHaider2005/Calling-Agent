# LATENCY_REPORT — honest measured numbers

Measured on this machine: NVIDIA MX330 (2 GB VRAM, CC 6.1), CUDA torch (cu126),
7.8 GB RAM, Python 3.12. Date: 2026-09-30. Raw data: `results/benchmark_results.json`.

## Before (CPU-only torch)

| Stage | Latency |
|---|---|
| VAD | ~1 ms |
| Brain | 0.1 ms |
| TTS (warm) | 1.3 s |
| STT (whisper-tiny, CPU) | 40–54 s |
| **End-to-end** | **42–62 s** |
| Peak VRAM | 0 MB (CPU-only) |
| Peak RAM | 601 MB |

## After (GPU — CUDA torch on MX330)

| Stage | Latency |
|---|---|
| VAD | ~1–13 ms |
| Brain | ~0 ms |
| TTS (warm) | 2.0–3.0 s |
| STT (whisper-tiny, GPU, warm) | 5.7–6.7 s |
| **End-to-end (warm)** | **7.7–9.7 s** |
| First call (cold) | 46.9 s (model + voice loading) |
| Peak VRAM | 277 MB |
| Peak RAM | ~20 MB (brain only) |

## Summary

- **GPU STT is ~6–9x faster than CPU** (5.7 s vs 40–54 s). This is the single
  biggest win from installing CUDA torch.
- End-to-end dropped from **42–62 s to 7.7–9.7 s** (after warmup).
- The **first call is slow (46.9 s)** because the STT model and TTS voice load on
  first use. Pre-warming both at startup would make even the first call fast.
- The ~1.5 s target is **still not met** — STT at ~5.7 s is the bottleneck.
  Reaching 1.5 s would need a smaller/faster STT or streaming partial results.

## Suggestions

1. **Pre-warm at startup**: load the STT model and TTS voices when the agent
   starts, so the first caller doesn't pay the 47 s cold-start cost.
2. **Faster STT**: a distilled/quantized model, or a streaming STT that emits
   partial results (the caller hears the reply sooner).
3. **Sentence-level TTS streaming**: start speaking the first sentence while
   synthesizing the rest (Piper supports chunked synthesis).
4. **GPU is now used**: 277 MB peak VRAM on the MX330 — well within the 2 GB limit.
