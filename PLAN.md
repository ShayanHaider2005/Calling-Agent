# PLAN — Calling-Agent overnight build

Goal: multilingual (EN/UR/HI/ES) speech-to-speech phone-call agent, chained pipeline:
EARS (Whisper STT) -> BRAIN (small LLM + state machine) -> MOUTH (TTS) -> CALL HANDLING (VAD, barge-in, language following, latency).

Hardware reality (measured): NVIDIA MX330, **2048 MiB VRAM** (task said 4GB; plan for 2GB), driver 581.42, CUDA 13.0. Python 3.12.10 via `py` launcher. $0 budget.

## Task list (priority order, timeboxed)

- [ ] **P0. Git + skeleton** (~60 min) — DONE FIRST
  - git init/remote/branch overnight-build (remote already had README; preserved)
  - folders: configs/ src/ scenarios/ tests/ scripts/ data/ checkpoints/
  - requirements.txt, .gitignore, scripts/check_env.py
  - First commit + first push
  - Acceptance: `py scripts/check_env.py` runs; branch pushed; no file >50MB staged.

- [ ] **P1. Ears evaluation + test-set tool** (~90 min)
  - src/eval.py: WER (jiwer) per language + per mix from metadata.csv, per-language normalization (Urdu/Hindi specifics), unit tests
  - scripts/baseline.py: run off-the-shelf small STT on test set -> baseline_results.json
  - scripts/record_testset.py: terminal tool, one sentence at a time, mic record on keypress (16kHz mono wav), redo/skip, appends metadata.csv
  - data/testset_prompts/: ~150 sentences (Urdu script, English, natural UR-EN mixed), marked UNVERIFIED
  - RECORDING_GUIDE.md
  - Acceptance: eval unit tests pass; baseline runs on at least a few clips; record tool records one clip.

- [ ] **P2. End-to-end pipeline** (~90 min)
  - src/pipeline.py: audio in -> VAD -> STT -> lang detect -> brain -> TTS -> audio out
  - mic + file/text input; barge-in; end-of-speech; per-stage latency log
  - Acceptance: text-mode and file-audio-mode runs end to end; latency log written.

- [ ] **P3. Brain + guardrails** (~90 min)
  - src/brain.py: state machine (greet, intent, FAQ, booking: name/day/time, confirm, handoff, end); LLM only for intent classification + light rephrase of approved answers; rule-based fallback with no LLM
  - scenarios/clinic.yaml (fictional, clearly fake)
  - tests/test_brain.py: simulated callers (interested, price, booking, lang switch UR<->EN, HI, ES, "are you human?", out-of-scope, not-offered, rude, "stop", trick/promise). >=50 simulated calls, report pass rate + failures.
  - Acceptance: all rule tests pass; >=50 calls simulated; pass rate reported.

- [ ] **P4. Mouth (TTS)** (~90 min)
  - Survey free TTS for EN/UR/HI/ES; LICENSES.md with license + commercial-use status
  - src/tts.py: common interface, per-language voice config, CPU
  - Intelligibility check: synth test sentences -> STT back -> WER -> tts_check.json; TTS_NOTES.md
  - Acceptance: tts.py synthesizes all 4 languages; tts_check.json written; TTS_NOTES.md honest.

- [ ] **P5. Latency + resources** (~60 min)
  - scripts/benchmark.py: per-stage timing, end-to-end delay, peak VRAM/RAM
  - LATENCY_REPORT.md with honest numbers
  - Acceptance: benchmark runs; report written with real measured numbers.

- [ ] **P6. Overnight training loop** (~90 min)
  - engine.py: `python engine.py night --hours N` — LoRA fine-tune small STT in time-budgeted rounds, checkpoints, eval per round, keep best, CSV log, rollback if worse, resume after interruption; STOP-file kill switch
  - Resume proof: test kills tiny run midway, restarts
  - Smoke test <200 short clips (permissive public dataset if no-login, else synthetic)
  - Acceptance: engine runs a tiny round; resume test passes; smoke eval produces scores.

- [ ] **P7. Local web demo** (~60 min)
  - localhost-only page: click to talk, hear reply, see transcript + latency
  - Acceptance: page serves; mic capture works in browser; audio plays back.

- [ ] **P8. Data intake dry-run** (~60 min)
  - scripts/ingest_links.py: links.txt -> segmentation, lang detect, transcription, filtering (captions match audio OR two outputs agree), dataset writing, license allowlist. NO video downloads tonight. Test on local sample audio only; unit-test filtering.
  - Acceptance: unit tests pass on synthetic segments; dry-run on local audio works.

- [ ] **P9. Docs** (~60 min)
  - README.md (install, record test set, run baseline, run agent, run overnight loop), DATA_SOURCES.md, DECISIONS.md
  - Acceptance: README commands all work.

## Final 30 min
- Final push of overnight-build; MORNING_REPORT.md (done/not done, push status + branch, assumptions, test results, TTS notes, latency/VRAM, license flags, problems, first 3 commands).

## Rules
- Never commit/push main; never force-push; no file >50MB; no secrets/recordings in git.
- Log blockers in PROGRESS.md under BLOCKED after 3 failed attempts / 20 min.
- Never claim untested results.
