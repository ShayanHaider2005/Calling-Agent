# PLAN — Calling-Agent session 3

Goal: a proper call UI for a test call tomorrow, a harder curriculum training run
that genuinely improves the ears, and a client-ready report with honest numbers.

Hardware (measured): NVIDIA MX330 **2048 MiB VRAM** (2GB), 7.8GB RAM, Python 3.12.10.
GPU is free (no leftover processes). $0 budget.

## GPU budget (hard cap)

- Total VRAM: **2048 MiB**. Hard working cap: **1.6 GB** (leave ~400 MB for OS/display).
- Enforced in code via `torch.cuda.set_per_process_memory_fraction(0.8)`.
- STT model on GPU; brain LLM + TTS on CPU.
- Catch OOM: lower batch size / segment length, clear cache, retry, log it.
- Target STT: **whisper-small** (244M) in 8-bit + LoRA + gradient checkpointing + batch 1.
  If it OOMs, fall back to whisper-base, then whisper-tiny. Log the choice.

## 5-hour schedule

- [ ] **R0. Orientation + GPU budget** (~30 min)
  - Run all tests; fix anything broken.
  - Verify GPU free; record total/free VRAM in DECISIONS.md.
  - Verify whisper-small fits under the cap (8-bit + LoRA); downgrade + log if not.
  - Acceptance: tests pass; model size chosen and recorded; cap enforced.

- [ ] **R1. Call UI** (~1.5 h, CPU)
  - `scripts/run_demo.py` — one command, localhost only.
  - Phone-call screen: Start/End Call, mic access, live waveform, who-is-speaking,
    live transcript, language badge, per-turn latency, mute.
  - Banner: "You are speaking with an AI assistant. This call may be recorded."
  - Barge-in; scenario selector (clinic/restaurant); language hint (auto/EN/UR/HI).
  - Text-input fallback; save call (transcript + latency + optional audio, git-ignored).
  - Status panel (models, sizes, DEMO-ONLY flags, CPU/GPU mode, VRAM).
  - `scripts/check_demo.py` — automated health check with synthetic caller.
  - RTL for Urdu; Urdu/Hindi font; low-memory mode (CPU STT).
  - Acceptance: UI runs; check_demo.py passes; within VRAM cap.

- [ ] **R2. Curriculum training** (~2.5 h, GPU)
  - Staged difficulty: clean -> noise -> phone-8kHz -> speed/volume -> mixed -> hard-example mining.
  - Advance only when dev score stops improving; keep best; roll back if worse.
  - Regression check: English must not get worse while Urdu improves.
  - Watch for overfitting; respect STOP; monitor every ~30 min.
  - Save scores CSV + plot (baseline vs each stage).
  - Acceptance: stages run; scores recorded; best checkpoint saved.

- [ ] **R3. Brain hardening** (~30 min, CPU)
  - Harder simulated callers: interrupting, vague, angry, mid-sentence switch, STT errors.
  - Target >=200 simulated calls; fix failures; report pass rate + ALL failures.
  - Acceptance: >=200 calls; pass rate + failures reported.

- [ ] **R4. Final measurements** (~30 min, GPU free after training)
  - Baseline vs final WER per language + per mix on held-out set.
  - End-to-end latency (GPU + CPU); peak VRAM/RAM; simulated-call pass rate; TTS check.
  - Acceptance: all measured and recorded.

- [ ] **R5. Client report** — CLIENT_REPORT.md (plain words, honest numbers, limitations, licensing, next steps).
- [ ] **R6. Demo guide** — DEMO_GUIDE.md (start UI, checklist, 10 test calls, weak spots, GPU freeing, failure recovery).
- [ ] **R7. README** — first person, UI instructions, GPU requirements, latest results.

## Final 30 min
- Final push; MORNING_REPORT_3.md (done/not done, push status, UI command, STT size + VRAM, training stages, pass rate + failures, latency, license flags, client-report cautions, problems, next 3 commands).

## Rules
- Work on overnight-build-3; never commit/push main; never force-push; no file >50MB.
- No two GPU jobs at once; UI/tests on CPU while training runs.
- Log blockers in PROGRESS.md under BLOCKED after 3 failed attempts / 20 min.
- Never claim untested results.
