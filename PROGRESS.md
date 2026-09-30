# PROGRESS — Calling-Agent overnight build

Format: newest first. All state lives here (memory may be reset).

## Session log

### 2026-09-30 06:15 — Start
- Environment measured: Python 3.12.10 (via `py` launcher; `python` is MS Store stub), git 2.51.1, NVIDIA MX330 **2048 MiB VRAM** (task assumed 4GB — planning for 2GB), driver 581.42, CUDA 13.0.
- Git: repo already initialized, origin = https://github.com/ShayanHaider2005/Calling-Agent.git (correct). Remote main has 1 commit (placeholder README.md) — preserved. Created branch `overnight-build` from origin/main.
- Venv: `.venv` being created (background).
- PLAN.md written.

## Tasks

- [x] P0a: git connection + branch (overnight-build from origin/main)
- [x] P0b: skeleton + check_env.py (venv, requirements, .gitignore, folders)
- [x] P1: src/eval.py + tests (9 pass), scripts/baseline.py, scripts/record_testset.py, 150 prompts, RECORDING_GUIDE.md
- [x] P3: src/brain.py + scenarios/clinic.yaml + tests/test_brain.py (18 pass, bulk 56/56 = 100%)
- [x] P2: src/pipeline.py (text/file/mic modes, VAD, barge-in, per-stage latency log) — text + file verified end-to-end
- [x] P4: src/tts.py (piper, all 4 languages verified) — intelligibility check pending
- [x] P5: scripts/benchmark.py + LATENCY_REPORT.md (honest numbers: brain 0.1ms, TTS 1.3s warm, STT 40-54s CPU bottleneck, RAM 601MB, VRAM 0)
- [ ] P6: training loop
- [ ] P7: web demo
- [ ] P8: data intake
- [ ] P9: docs

## Test results so far
- tests/test_eval.py: 9 passed
- tests/test_brain.py: 18 passed; bulk simulation 56/56 (100%)
- scripts/check_env.py: runs; python_ok, gpu_ok (CPU-only torch), disk_ok; libs_ok=False only because check_env looks for "TTS" which we replaced with piper-tts (cosmetic)

## BLOCKED
(none yet)

## Notes / decisions so far
- GPU is 2GB (MX330), not 4GB: using whisper-tiny/small on CPU, rule-based brain (no LLM by default), CPU TTS. Measure and record real peaks.
- RAM is 7.8 GB total, ~1.6 GB free: LLM (1.5B) is optional/off by default; rule-based fallback is the tested path.
- torch installed as CPU-only (2.14.0+cpu) to stay under the 6GB download budget (CUDA wheels + nvidia libs ~4GB). STT runs on CPU.
- Coqui TTS has no Windows wheels -> replaced with piper-tts (MIT, local, CPU) + pyttsx3/SAPI5 fallback.
- check_env.py lists "TTS" as missing — cosmetic; requirements use piper-tts instead.
