# RECORDING_GUIDE — test-set audio

The test set is used **only for evaluation** (measuring WER). It is **NEVER used for training**.
Keep it small, clean, and out of git (recordings are git-ignored).

## How to record

```
py scripts/record_testset.py              # all prompts (urdu, english, mixed)
py scripts/record_testset.py --lang urdu  # one language at a time
```

Flow per sentence:
1. The sentence is shown on screen.
2. Press **Enter** to start recording, read the sentence naturally, press **Enter** to stop.
3. Press **r** to redo, **s** to skip, **q** to quit (progress is saved).

Output: `data/testset_audio/<lang>_<idx>.wav` (16 kHz mono) + a row in `data/metadata.csv`.

## Tips for good data

- Record in a quiet room, ~15–30 cm from the mic, normal speaking volume.
- Read naturally — do not over-pronounce. Small pauses are fine.
- For **mixed** sentences, code-switch the way you actually would on a phone call.
- Aim for at least 30–50 clips per language; ~150 total is the target.
- If a sentence feels unnatural, skip it and note it in PROGRESS.md.

## Phone-quality samples (optional but valuable)

To measure real-world performance, also record ~10 clips that sound like a phone call:
- Speak slightly quieter and farther from the mic.
- If you can, record through a phone speaker playing the prompt (or use a low-quality
  laptop mic / headset). Name them `phone_<lang>_<idx>.wav` and add rows to
  `metadata.csv` with the same columns. Mark them with language `phone_english` etc.
  so you can score them separately.

## Rules

- Only **you** may be recorded. Never record anyone else.
- The set is never used for training — it measures the model, it does not teach it.
- Prompts in `data/testset_prompts/` are **UNVERIFIED** until you check them.
