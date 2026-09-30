# MORNING_REPORT — Calling-Agent overnight build

Date: 2026-09-30. Branch: **overnight-build** (pushed to origin). All work is on
`overnight-build`; `main` was never touched. The original remote `main` commit
(75987db "Initial commit") is preserved at the base of the branch.

## GitHub push

**Worked.** Every commit was pushed to `https://github.com/ShayanHaider2005/Calling-Agent`
on branch `overnight-build`. 8 commits total. No authentication issues, no force-push,
no history rewrite. `git status` is clean and `origin/overnight-build` matches local.

## What was completed (P0–P9, all done)

- **P0 Git + skeleton**: git init/remote/branch, folders, requirements.txt, .gitignore,
  `scripts/check_env.py`. First commit + push.
- **P1 Ears eval + test-set tool**: `src/eval.py` (per-language WER with Urdu/Hindi
  normalization) + 9 unit tests; `scripts/baseline.py`; `scripts/record_testset.py`;
  150 test-set prompts (UNVERIFIED); RECORDING_GUIDE.md.
- **P2 Pipeline**: `src/pipeline.py` — text/file/mic modes, energy VAD, barge-in,
  per-stage latency log. Text + file verified end-to-end.
- **P3 Brain + guardrails**: `src/brain.py` (state machine, rule-based intent,
  language following, guardrails), `scenarios/clinic.yaml` (fictional clinic),
  `tests/test_brain.py` — **18 tests pass, bulk simulation 56/56 (100%)**.
- **P4 Mouth (TTS)**: `src/tts.py` (Piper, all 4 languages verified), intelligibility
  check, TTS_NOTES.md.
- **P5 Latency + resources**: `scripts/benchmark.py`, LATENCY_REPORT.md (honest numbers).
- **P6 Overnight training loop**: `engine.py` (LoRA night/smoke/eval), `tests/test_engine.py`
  — **4 tests pass** (smoke, resume-continues, STOP-file, best-checkpoint).
- **P7 Web demo**: `scripts/web_demo.py` (FastAPI, localhost) — verified GET / 200,
  POST /chat 200.
- **P8 Data intake dry-run**: `scripts/ingest_links.py` + `tests/test_ingest.py` —
  **6 tests pass**. No video downloads (as required).
- **P9 Docs**: README.md, DATA_SOURCES.md, DECISIONS.md, LICENSES.md.

## Test results (all run and observed)

| Suite | Result |
|---|---|
| tests/test_eval.py | 9 passed |
| tests/test_brain.py | 18 passed; **bulk simulation 56/56 (100%)** |
| tests/test_ingest.py | 6 passed |
| tests/test_engine.py | 4 passed (smoke, resume, STOP, best-ckpt) |
| **total** | **37 passed, 0 failed** |

Simulated-call pass rate: **100% (56/56)**. The 56 calls cover: FAQ (4 languages),
trick/promise questions, out-of-scope, not-offered, rude callers, booking flows,
language switches (UR<->EN, HI, ES), identity/human requests, "stop", cancel/reschedule,
and mixed-language utterances. No failures in the final run.

## TTS quality per language (round-trip WER: TTS -> STT)

| Language | WER | Verdict |
|---|---|---|
| English | 0.20 | Acceptable (the 0.20 is "9 AM" vs "9am" casing, not intelligibility) |
| Spanish | 0.48 | Understandable but weak; needs native-speaker listening test |
| Urdu | 0.78 | **Weak** — needs a better voice + native-speaker test |
| Hindi | 1.00 | **Poor / DEMO-ONLY** — STT hears romanized garbage, not Devanagari speech |

## Latency and resources (measured, honest)

- Brain: **0.1 ms** mean (rule-based, no LLM). RAM 19 MB.
- TTS (Piper): **1.3 s** per sentence after warmup (4-6 s first call).
- VAD: ~1 ms.
- STT (whisper-tiny, CPU): **40-54 s** on a synthetic tone (runaway dot-generation);
  real speech would be faster but still ~15-30 s on CPU.
- **End-to-end: 42-62 s. The ~1.5 s target is NOT met on CPU.** STT is the bottleneck.
- Peak VRAM: **0 MB** (CPU-only torch; the MX330 is unused).
- Peak RAM: **601 MB** (STT + TTS + brain loaded).

## License flags

- whisper-tiny/small: MIT — commercial OK.
- Piper + voices: MIT/CC0 — commercial OK.
- Qwen2.5-1.5B-Instruct (optional LLM): Apache-2.0 — commercial OK.
- Coqui TTS: surveyed, **not used** (no Windows wheels).
- **Hindi TTS: DEMO-ONLY** (weak voice). Urdu TTS: weak, verify before real use.
- Test-set recordings: user-owned, evaluation only, never for training.

## What was NOT completed / known gaps

1. **CPU STT is too slow** for the ~1.5 s target. Fix: GPU STT (CUDA torch + whisper
   on the MX330). Not done: CUDA wheels ~4 GB (download budget) + 2 GB VRAM.
2. **Urdu/Hindi TTS voices are weak.** Need better voices + native-speaker listening
   tests before any real use.
3. **Test-set prompts are UNVERIFIED** — need the user to check the 150 sentences.
4. **No real recordings yet** — `data/metadata.csv` doesn't exist until the user runs
   `record_testset.py`. So `baseline.py` and the training eval return NaN WER.
5. **No real phone lines** (out of scope for tonight).
6. **The optional LLM path is untested at runtime** (rule-based brain is the tested
   default; the LLM is wired but off by default due to RAM).

## Assumptions made (also in DECISIONS.md)

- GPU is 2 GB (MX330), not 4 GB — planned conservatively.
- CPU-only torch to stay under the 6 GB download budget.
- whisper-tiny (not small) to fit RAM.
- Rule-based brain by default (LLM optional/off) due to ~1.6 GB free RAM.
- Piper for TTS (Coqui has no Windows wheels).
- Synthetic data for the training smoke test (public dummy dataset blocked by
  ffmpeg + broken file paths).
- Energy-based VAD (not webrtcvad) for portability.
- Audio as numpy arrays to STT (no ffmpeg for file reads).

## Problems encountered and fixed

- Coqui TTS: no Windows wheels -> switched to Piper.
- transformers ASR pipeline needs ffmpeg for file paths -> pass numpy arrays instead.
- Devanagari virama / Urdu yeh killed by generic punctuation regex -> script-aware
  normalization.
- peft `task_type='SEQ_2_SEQ_LM'` conflicts with transformers 5.17 -> removed it.
- Whisper mel features need padding to length 3000 -> added in collate.
- Saved LoRA adapters default to `inference_mode=True` -> load with `is_trainable=True`.
- STOP-file race with round completion -> check STOP after each round.
- Windows subprocess output decoding -> UTF-8 in tests.

## The exact first three commands to run

```bash
# 1. Record a few test-set clips (optional, for evaluation)
.venv\Scripts\python.exe scripts\record_testset.py

# 2. Run the agent demo (text mode, no mic needed)
.venv\Scripts\python.exe -m src.pipeline text

# 3. Run the overnight training loop (tiny smoke first, then a real night)
.venv\Scripts\python.exe engine.py smoke
.venv\Scripts\python.exe engine.py night --hours 4
```

(If the venv isn't active, use the full path `.venv\Scripts\python.exe`.)

## Bottom line

The full chained pipeline works end to end and is tested (37 tests pass, 56/56
simulated calls). The brain follows the rules. The two things that need your attention
before this is "real": **record the test set** (so WER is meaningful) and **GPU STT**
(so latency is acceptable). Everything else is documented and ready to build on.
