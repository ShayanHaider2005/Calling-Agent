"""Release gate: a new language pack must improve its own language on the dev
set AND not regress the other languages beyond a set tolerance.

Usage:
  py scripts/release_gate.py --language english --adapter checkpoints/best
  py scripts/release_gate.py --language urdu --adapter checkpoints/urdu_best

Evaluates per-language WER on the dev set (data/devset_metadata.csv) with and
without the pack's LoRA adapter. A pack PASSES if:
  - its own language WER improves (or stays within tolerance), AND
  - no other language regresses by more than REGRESSION_TOLERANCE.

Languages with no dev data are skipped (reported as "no data").

Exit code 0 = pass, 1 = fail.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval import evaluate

# Max allowed WER increase for other languages (absolute, e.g. 0.05 = 5 points).
REGRESSION_TOLERANCE = 0.05
# Min required WER improvement for the pack's own language (absolute).
IMPROVEMENT_THRESHOLD = 0.0


def load_metadata(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def eval_wer(rows, model, processor):
    """Evaluate WER for a model on the given rows. Returns {lang: wer}."""
    import soundfile as sf
    import torch
    from src.eval import normalize_text, compute_wer

    per_lang = {}
    by_lang = {}
    for row in rows:
        lang = row.get("language", "english").lower()
        by_lang.setdefault(lang, {"refs": [], "hyps": []})
        path = row["path"]
        if not os.path.isabs(path):
            path = os.path.join(".", path)
        if not os.path.exists(path):
            by_lang[lang]["refs"].append(normalize_text(row.get("text", ""), lang))
            by_lang[lang]["hyps"].append("")
            continue
        pcm, sr = sf.read(path, dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        inp = processor(pcm, sampling_rate=sr, return_tensors="pt")
        inp = {k: v.to(model.device) for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp)
        hyp = processor.batch_decode(out, skip_special_tokens=True)[0]
        by_lang[lang]["refs"].append(normalize_text(row.get("text", ""), lang))
        by_lang[lang]["hyps"].append(normalize_text(hyp, lang))
    for lang, d in by_lang.items():
        wer, _ = compute_wer(d["refs"], d["hyps"])
        per_lang[lang] = wer
    return per_lang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, help="pack language (english|urdu|hindi)")
    ap.add_argument("--adapter", default=None, help="path to the pack's LoRA adapter")
    ap.add_argument("--metadata", default="data/devset_metadata.csv")
    ap.add_argument("--model", default="openai/whisper-tiny")
    ap.add_argument("--regression-tolerance", type=float, default=REGRESSION_TOLERANCE)
    args = ap.parse_args()

    if not os.path.exists(args.metadata):
        print(f"no dev set at {args.metadata} — cannot run release gate")
        sys.exit(1)

    rows = load_metadata(args.metadata)
    langs_in_data = sorted({r.get("language", "english").lower() for r in rows})
    print(f"dev set: {len(rows)} clips, languages: {langs_in_data}")
    print(f"pack: language={args.language} adapter={args.adapter}")

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(args.model)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model)

    # baseline (no adapter)
    base_wer = eval_wer(rows, base_model, processor)
    print(f"\nbaseline WER (no adapter): {base_wer}")

    # with adapter
    if args.adapter and os.path.exists(args.adapter):
        model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
        pack_wer = eval_wer(rows, model, processor)
        print(f"pack WER (with adapter):  {pack_wer}")
    else:
        print(f"adapter not found at {args.adapter} — using baseline as pack WER")
        pack_wer = dict(base_wer)

    # gate check
    print("\n--- release gate ---")
    passed = True
    # own language must improve (or stay within tolerance)
    own = args.language
    if own in pack_wer and own in base_wer:
        delta = pack_wer[own] - base_wer[own]
        ok = delta <= IMPROVEMENT_THRESHOLD
        print(f"  {own}: base={base_wer[own]:.4f} pack={pack_wer[own]:.4f} delta={delta:+.4f} -> {'OK' if ok else 'FAIL'}")
        passed = passed and ok
    else:
        print(f"  {own}: no dev data — skipped")
    # other languages must not regress beyond tolerance
    for lang in langs_in_data:
        if lang == own:
            continue
        if lang in pack_wer and lang in base_wer:
            delta = pack_wer[lang] - base_wer[lang]
            ok = delta <= args.regression_tolerance
            print(f"  {lang}: base={base_wer[lang]:.4f} pack={pack_wer[lang]:.4f} delta={delta:+.4f} -> {'OK' if ok else 'FAIL'}")
            passed = passed and ok
        else:
            print(f"  {lang}: no dev data — skipped")

    print(f"\nrelease gate: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
