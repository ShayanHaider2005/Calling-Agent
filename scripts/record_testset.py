"""Terminal tool to record the test set from a microphone.

Shows one prompt sentence at a time. Press:
  [Enter]  start recording (records until you press Enter again)
  [r]      redo the current sentence
  [s]      skip the current sentence
  [q]      quit (progress is saved)

Recordings are 16 kHz mono WAV saved under data/testset_audio/ and appended to
data/metadata.csv with columns: path, language, text, is_mixed.

IMPORTANT: this set is NEVER used for training — only for evaluation.
Recordings stay out of git (see .gitignore).

Usage:
  py scripts/record_testset.py                 # all prompt files
  py scripts/record_testset.py --lang urdu    # only urdu.txt
"""
import argparse
import csv
import os
import sys
import time

import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_DIR = os.path.join("data", "testset_audio")
METADATA = os.path.join("data", "metadata.csv")
PROMPT_DIR = os.path.join("data", "testset_prompts")

FIELDS = ["path", "language", "text", "is_mixed"]


def load_prompts(lang_filter=None):
    """Return list of (language, sentence) from prompt txt files."""
    prompts = []
    files = {
        "urdu": "urdu.txt",
        "english": "english.txt",
        "mixed": "mixed.txt",
    }
    for lang, fname in files.items():
        if lang_filter and lang != lang_filter:
            continue
        path = os.path.join(PROMPT_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    prompts.append((lang, line))
    return prompts


def load_done(metadata_path):
    """Return set of (language, text) already recorded."""
    done = set()
    if os.path.exists(metadata_path):
        with open(metadata_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["language"], row["text"]))
    return done


def append_metadata(metadata_path, row):
    exists = os.path.exists(metadata_path)
    with open(metadata_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def record_until_key():
    """Record from mic until Enter is pressed. Returns numpy array or None."""
    print("  recording... press Enter to stop", flush=True)
    chunks = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [audio status: {status}]", file=sys.stderr)
        chunks.append(indata.copy())

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                dtype="int16", callback=callback)
    except Exception as e:
        print(f"  could not open microphone: {e}")
        return None

    with stream:
        while True:
            ch = input()
            if ch == "":
                break
    if not chunks:
        return None
    import numpy as np
    return np.concatenate(chunks, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default=None, help="urdu | english | mixed")
    ap.add_argument("--device", type=int, default=None, help="sounddevice device index")
    args = ap.parse_args()

    if args.device is not None:
        sd.default.device = args.device

    print("Calling-Agent test-set recorder")
    print(f"sample rate: {SAMPLE_RATE} Hz, channels: {CHANNELS}")
    print("Controls: Enter=start/stop recording, r=redo, s=skip, q=quit\n")

    prompts = load_prompts(args.lang)
    done = load_done(METADATA)
    todo = [(l, t) for (l, t) in prompts if (l, t) not in done]
    print(f"{len(prompts)} prompts total, {len(done)} already recorded, {len(todo)} to go\n")

    os.makedirs(AUDIO_DIR, exist_ok=True)
    recorded = 0

    for i, (lang, text) in enumerate(todo):
        print(f"[{i+1}/{len(todo)}] ({lang}) {text}")
        while True:
            key = input("  [Enter]=record  [r]=redo  [s]=skip  [q]=quit > ").strip().lower()
            if key == "q":
                print(f"\nquit. recorded {recorded} this session.")
                return
            if key == "s":
                print("  skipped.")
                break
            if key == "r":
                print("  redo — recording again.")
            # record
            audio = record_until_key()
            if audio is None or len(audio) < SAMPLE_RATE // 4:
                print("  too short or no audio — try again.")
                continue
            idx = len(done) + recorded
            fname = f"{lang}_{idx:03d}.wav"
            fpath = os.path.join(AUDIO_DIR, fname)
            sf.write(fpath, audio, SAMPLE_RATE)
            is_mixed = "1" if lang == "mixed" else "0"
            append_metadata(METADATA, {"path": fpath, "language": lang,
                                      "text": text, "is_mixed": is_mixed})
            done.add((lang, text))
            recorded += 1
            dur = len(audio) / SAMPLE_RATE
            print(f"  saved {fpath} ({dur:.1f}s)\n")
            break

    print(f"\ndone. recorded {recorded} clips this session.")


if __name__ == "__main__":
    main()
