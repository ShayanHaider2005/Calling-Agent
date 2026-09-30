# DATA_SOURCES — where training/eval data comes from

## Test set (evaluation only, NEVER for training)

- **Prompts**: `data/testset_prompts/{english,urdu,mixed}.txt` — 150 sentences
  (50 per language), marked **UNVERIFIED** until the user checks them.
- **Recordings**: `data/testset_audio/*.wav` (16 kHz mono) recorded with
  `scripts/record_testset.py`. Metadata in `data/metadata.csv`
  (columns: path, language, text, is_mixed).
- **Rule**: this set measures the model; it does **not** train it. See RECORDING_GUIDE.md.

## Training data (for the overnight LoRA loop)

The loop (`engine.py`) supports three training sources (`train_kind`):

1. **synthetic** (default for smoke test): piper-TTS clips with known transcripts.
   Self-contained, no ffmpeg, no downloads. Used to prove the loop works.
2. **librispeech_dummy**: `hf-internal-testing/librispeech_asr_dummy` (MIT test fixture).
   Surveyed but **blocked**: audio decode needs ffmpeg (torchcodec, not installed) and
   the `file` column holds the original creator's absolute paths (broken locally).
3. **local**: `data/train/` with wav files + `transcripts.csv` (path, text).
   This is the intended source for real training: record or curate clips, transcribe
   them, and point the loop at them.

### Intake (future, dry-run tonight)

`scripts/ingest_links.py` implements the future pipeline: read `links.txt` ->
license allowlist -> segment -> language detect -> transcribe -> filter
(captions match audio OR two outputs agree) -> write dataset. No video downloads
were made tonight. The filtering logic is unit-tested.

## What is needed for real training

- A few hundred to a few thousand short clips with accurate transcripts, per language.
- For Urdu/English mixed speech, natural code-switched clips.
- All data must be permissively licensed (see LICENSES.md) and never the test set.
