"""Extract a small public dev set from a cached HF dataset (bypasses ffmpeg).

Reads the raw audio bytes directly from the pyarrow table (no torchcodec), writes
16 kHz mono WAV files, and creates a metadata.csv. Used to build a labelled
"public dev set" stand-in when no user test set exists.

Usage:
  py scripts/extract_devset.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    from datasets import load_dataset
    import soundfile as sf
    import numpy as np

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation")
    table = ds.data
    audio_col = table.column("audio")
    text_col = table.column("text")

    out_dir = os.path.join("data", "devset_audio")
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join("data", "devset_metadata.csv")

    n = min(20, len(ds))
    rows = []
    for i in range(n):
        a = audio_col[i].as_py()
        b = a.get("bytes") if a else None
        if not b:
            continue
        # write raw bytes (flac) then read with soundfile
        import io
        pcm, sr = sf.read(io.BytesIO(b), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        if sr != 16000:
            pcm = _resample(pcm, sr, 16000)
        fname = f"dev_{i:03d}.wav"
        fpath = os.path.join(out_dir, fname)
        sf.write(fpath, pcm, 16000)
        text = text_col[i].as_py()
        rows.append({"path": fpath, "language": "english", "text": text,
                     "is_mixed": "0"})
        print(f"  {fname}: {text!r}")

    import csv
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "language", "text", "is_mixed"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} clips to {out_dir}")
    print(f"wrote {meta_path}")
    print("LABEL: public dev set (hf-internal-testing/librispeech_asr_dummy), NOT the user's test set")


def _resample(pcm, orig_sr, target_sr):
    if orig_sr == target_sr:
        return pcm
    import math
    gcd = math.gcd(orig_sr, target_sr)
    n_out = int(len(pcm) * target_sr / orig_sr)
    x_old = [i / (len(pcm) - 1) for i in range(len(pcm))] if len(pcm) > 1 else [0]
    x_new = [i / (n_out - 1) for i in range(n_out)] if n_out > 1 else [0]
    return np.interp(x_new, x_old, pcm).astype(np.float32)


if __name__ == "__main__":
    main()
