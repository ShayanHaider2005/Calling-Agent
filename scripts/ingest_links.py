"""Data intake (DRY-RUN ONLY — no video downloads tonight).

Reads links.txt (video URLs + optional license), and for each link would:
  1. check the license against an allowlist (LICENSE_ALLOWLIST)
  2. segment the audio into utterances
  3. detect language per segment
  4. transcribe each segment
  5. filter: keep a segment if (captions match the audio) OR (two transcription
     outputs agree) — i.e. trust the audio when captions and audio disagree
  6. write the dataset (audio + transcript + language + source + license)

Tonight: NO downloads. The pipeline functions are implemented and unit-tested on
local sample audio. `py scripts/ingest_links.py --dry-run` processes a local audio
file through the same code path (segment -> lang detect -> transcribe -> filter).

Usage:
  py scripts/ingest_links.py --dry-run --audio data/sample.wav
  py scripts/ingest_links.py --links links.txt   (future: real download)
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.brain import detect_language
from src.eval import normalize_text, compute_wer

# Licenses we are allowed to train on (commercially usable, permissive).
LICENSE_ALLOWLIST = {
    "CC0", "CC-BY", "CC-BY-SA", "MIT", "Apache-2.0", "Unlicense",
    "CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "CC-BY-SA-4.0",
}

# WER threshold: below this, two outputs "agree" / captions "match" the audio.
MATCH_WER_THRESHOLD = 0.30


# ------------------------------------------------------------------ pipeline steps

def check_license(license_str):
    """Return True if the license is in the allowlist."""
    if not license_str:
        return False
    return license_str.strip().upper() in {l.upper() for l in LICENSE_ALLOWLIST}


def segment_audio(pcm, sample_rate=16000, max_seconds=30):
    """Split audio into segments on silence. Returns list of (start, end) samples.

    Simple energy-based segmentation (dry-run quality). A real implementation
    would use a proper VAD + speaker diarization.
    """
    from src.pipeline import EnergyVAD
    vad = EnergyVAD(sample_rate=sample_rate, silence_ms=500)
    frame_len = vad.frame_len
    n = len(pcm)
    n_frames = n // frame_len
    if n_frames == 0:
        return [(0, n)]
    segments = []
    in_speech = False
    start = 0
    silence_run = 0
    for i in range(n_frames):
        chunk = pcm[i * frame_len:(i + 1) * frame_len]
        if vad.is_speech(chunk):
            if not in_speech:
                start = i * frame_len
                in_speech = True
            silence_run = 0
        else:
            if in_speech:
                silence_run += 1
                if silence_run >= vad.silence_frames:
                    end = (i - silence_run + 1) * frame_len
                    segments.append((start, end))
                    in_speech = False
                    silence_run = 0
    if in_speech:
        segments.append((start, n))
    # cap segment length at max_seconds
    capped = []
    for s, e in segments:
        max_len = int(max_seconds * sample_rate)
        while e - s > max_len:
            capped.append((s, s + max_len))
            s += max_len
        capped.append((s, e))
    return capped


def transcribe_segment(pcm, sample_rate, stt):
    """Transcribe one audio segment. Returns the transcript text."""
    return stt.transcribe_array(pcm, sample_rate)


def captions_match_audio(captions_text, audio_transcript, language):
    """True if the captions and the audio transcript agree (WER below threshold)."""
    ref = normalize_text(captions_text, language)
    hyp = normalize_text(audio_transcript, language)
    if not ref:
        return False
    wer, _ = compute_wer([ref], [hyp])
    return wer <= MATCH_WER_THRESHOLD


def two_outputs_agree(output_a, output_b, language):
    """True if two transcription outputs agree (WER below threshold)."""
    a = normalize_text(output_a, language)
    b = normalize_text(output_b, language)
    if not a or not b:
        return False
    wer, _ = compute_wer([a], [b])
    return wer <= MATCH_WER_THRESHOLD


def filter_segment(captions_text, audio_transcript, second_transcript, language):
    """Keep a segment if captions match the audio OR two outputs agree.

    Returns (keep: bool, reason: str).
    """
    if captions_match_audio(captions_text, audio_transcript, language):
        return True, "captions_match_audio"
    if two_outputs_agree(audio_transcript, second_transcript, language):
        return True, "two_outputs_agree"
    return False, "rejected"


def write_dataset(rows, out_path):
    """Write the accepted segments to a dataset CSV."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "text", "language", "source", "license", "filter_reason"])
        for r in rows:
            w.writerow([r["path"], r["text"], r["language"], r["source"],
                        r["license"], r["filter_reason"]])


# ------------------------------------------------------------------ dry run

def dry_run(audio_path):
    """Run the intake pipeline on a local audio file (no download)."""
    import soundfile as sf
    from src.pipeline import WhisperSTT

    pcm, sr = sf.read(audio_path, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    print(f"audio: {audio_path} ({len(pcm)/sr:.1f}s, {sr} Hz)")

    segments = segment_audio(pcm, sr)
    print(f"segments: {len(segments)}")

    stt = WhisperSTT()
    rows = []
    for i, (s, e) in enumerate(segments):
        seg = pcm[s:e]
        transcript = transcribe_segment(seg, sr, stt)
        lang = detect_language(transcript)
        # dry-run: no captions, no second output -> filter rejects (nothing to match)
        keep, reason = filter_segment("", transcript, "", lang)
        print(f"  seg {i}: lang={lang} transcript={transcript!r} -> {reason}")
        rows.append({"path": f"{audio_path}#{i}", "text": transcript,
                     "language": lang, "source": "dry-run", "license": "",
                     "filter_reason": reason})
    write_dataset(rows, "results/intake_dryrun.csv")
    print(f"wrote results/intake_dryrun.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--links", default="links.txt")
    args = ap.parse_args()

    if args.dry_run:
        if not args.audio:
            print("--dry-run needs --audio <wav>")
            sys.exit(1)
        dry_run(args.audio)
        return

    # future: real download path (NOT implemented tonight)
    print("Real download is not implemented (dry-run only tonight).")
    print("links.txt format: url,license  (one per line)")


if __name__ == "__main__":
    main()
