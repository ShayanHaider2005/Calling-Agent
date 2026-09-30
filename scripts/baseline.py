"""Run the off-the-shelf small Whisper STT model on the test set and save baseline WER.

Usage:
  py scripts/baseline.py                      # full test set, whisper-small
  py scripts/baseline.py --model openai/whisper-tiny --limit 10
  py scripts/baseline.py --device cpu

Saves results/baseline_results.json with per-language and mixed/pure WER.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval import evaluate


def load_metadata(path):
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--device", default=None, help="cpu, cuda, or cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/baseline_results.json")
    args = ap.parse_args()

    rows = load_metadata(args.metadata)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no rows in metadata")
        sys.exit(1)

    import torch
    from transformers import pipeline

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={args.model} device={device} n={len(rows)}")

    asr = pipeline("automatic-speech-recognition", model=args.model,
                   device=0 if device.startswith("cuda") else -1)

    hypotheses = {}
    t0 = time.time()
    for i, row in enumerate(rows):
        path = row["path"]
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), "..", path)
        try:
            out = asr(path, generate_kwargs={"language": row.get("language", "english")})
            hypotheses[row["path"]] = out["text"]
        except Exception as e:
            print(f"  [{i+1}/{len(rows)}] ERROR {path}: {e}")
            hypotheses[row["path"]] = ""
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(rows)}] {time.time()-t0:.1f}s elapsed")

    result = evaluate(args.metadata if not args.limit else _write_subset(rows), hypotheses)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "device": device,
                   "wer": result}, f, indent=2, ensure_ascii=False)
    summary = {k: v for k, v in result.items() if k != "details"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


def _write_subset(rows):
    """Write a subset metadata file so evaluate() only scores the limited rows."""
    import csv
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="metadata_subset_")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "language", "text", "is_mixed"])
        w.writeheader()
        w.writerows(rows)
    return path


if __name__ == "__main__":
    main()
