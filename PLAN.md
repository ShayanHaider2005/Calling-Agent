# PLAN — Calling-Agent session 2

Goal: multilingual (English / Urdu / Hindi) speech-to-speech phone-call agent.
**Three languages only — Spanish is removed.** Chained pipeline:
EARS (Whisper STT) -> BRAIN (small LM + state machine) -> MOUTH (TTS) -> CALL HANDLING (VAD, barge-in, language following, latency).

Hardware (measured): NVIDIA MX330 **2048 MiB VRAM** (task says 4GB; plan for 2GB), 7.8GB RAM (~1.5GB free), Python 3.12.10 via `py`. $0 budget.

## Task list (priority order, timeboxed)

- [ ] **Q0. Orientation + repair** (~60 min)
  - Remove Spanish from all code/tests/docs (test sentences, simulated callers, TTS survey, config)
  - Run all tests; fix anything broken
  - Solve session-1 gaps that fit the constraints
  - Acceptance: 3 languages only; all tests pass.

- [ ] **Q1. Test data check** (~60 min)
  - Look for `data/testset` (metadata.csv + audio). If a verified set exists, use it; run baseline.py if baseline_results.json missing.
  - If NOT: use a small public dev split clearly labelled "public dev set, not my test set" as a temporary stand-in. Never train on it.
  - Acceptance: a labelled dev set exists; baseline runs if real test set present.

- [ ] **Q2. Real training run** (~5 h, background)
  - `python engine.py night` in background.
  - Data: public permissively-licensed speech (no login) + augmentation (noise, 8kHz phone sim) + synthetic mixed-language (labelled).
  - Monitor ~every 30 min; respect STOP; rollback if worse; save best. CPU work only while GPU busy.
  - Acceptance: training runs; score progression logged.

- [ ] **Q3. Language packs** (~90 min)
  - Per-language adapter loading + language-detection routing.
  - `scripts/release_gate.py`: a new pack must improve its own language on test/dev AND not regress others beyond tolerance.
  - Hindi pack skeleton; smoke-train only if time.
  - Acceptance: release_gate.py works; Hindi pack skeleton exists.

- [ ] **Q4. Brain** (~90 min)
  - Fix session-1 simulated-call failures.
  - Expand to 150 simulated calls: UR-EN and HI-EN switching, disfluencies, injected STT errors, unclear input (agent asks to repeat).
  - Second demo scenario (restaurant reservations) to prove business data is swappable.
  - Report pass rate + all failures.

- [ ] **Q5. Latency** (~60 min)
  - Optimize: faster inference engine (if license allows), quantization, warm-up, sentence-level TTS streaming.
  - Report before/after honestly.

- [ ] **Q6. Video-link data intake** — only if links.txt has allowlisted (CC/public-domain) entries; else skip.

- [ ] **Q7. Docs + CI** (~60 min)
  - Rewrite README (first person, Mermaid diagram, honest status, real results table, limitations, responsible use, roadmap).
  - GitHub Actions workflow: CPU-only unit tests (no model downloads, no cost).

## Final 30 min
- Final push; MORNING_REPORT_2.md (done/not done, push status, test set vs stand-in, score progression, pass rate + failures, latency before/after, license flags, problems, next 3 commands).

## Rules
- Work on the current development branch; do NOT create new branches (per user instruction).
- Never commit/push to main/master; never force-push; no file >50MB; no secrets/recordings in git.
- Log blockers in PROGRESS.md under BLOCKED after 3 failed attempts / 20 min.
- Never claim untested results.
