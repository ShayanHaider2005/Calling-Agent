"""Word error rate (WER) evaluation for the STT test set.

Reads metadata.csv with columns: path, language, text, is_mixed
  - path:     wav file path (relative to project root or absolute)
  - language: english | urdu | hindi | spanish
  - text:     reference transcript
  - is_mixed: 1 if the utterance mixes languages, else 0

Computes per-language WER, mixed-vs-pure WER, and overall WER using jiwer,
with per-language text normalization (Urdu/Hindi/Spanish specifics).

Usage:
  py src/eval.py --metadata data/metadata.csv --hypotheses results/hyp.json
  py src/eval.py --metadata data/metadata.csv --hypotheses results/hyp.json --out results/wer.json
"""
import argparse
import csv
import json
import re
import unicodedata

import jiwer

# ---------------------------------------------------------------- normalization

# Arabic-script letter variants that should be treated as equivalent.
_URDU_VARIANTS = {
    "آ": "ا",  # alif madda -> alif
    "أ": "ا",  # alif hamza above -> alif
    "إ": "ا",  # alif hamza below -> alif
    "ة": "ه",  # ta marbuta -> ha
    "ى": "ي",  # alif maksura -> ya
    "ہ": "ه",  # gol ha -> ha (Urdu)
    "ک": "ك",  # keheh -> kaf (Urdu)
    "ی": "ي",  # yeh (Urdu/Arabic) -> yeh
    "ے": "ي",  # bari yeh -> yeh
    "ؤ": "و",  # waw hamza -> waw
    "ئ": "ي",  # yeh hamza -> yeh
    "ٹ": "ت",  # tt -> t (retroflex collapse)
    "ڈ": "د",  # dd -> d
    "ڑ": "ر",  # rr -> r
    "ژ": "ز",  # zhe -> z
    "پ": "پ",  # keep
    "چ": "چ",  # keep
    "گ": "گ",  # keep
    "ں": "ن",  # noon ghunna -> n
}

# Devanagari (Hindi) normalization: nukta variants, danda, matra collapse.
_HINDI_VARIANTS = {
    "।": " ",  # danda -> space
    "ड़": "ड़",
    "ढ़": "ढ़",
}

_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D4-\u06ED\u0640]")  # Arabic harakat, tatweel
# Punctuation removal that PRESERVES script letters/marks (so conjuncts survive).
_PUNCT_AR = re.compile(r"[^\w\s\u0600-\u06FF]", re.UNICODE)   # keep Arabic block
_PUNCT_DEV = re.compile(r"[^\w\s\u0900-\u097F]", re.UNICODE)  # keep Devanagari block
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(text, language):
    """Normalize a transcript for WER comparison, per language."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    lang = (language or "").lower()

    if lang in ("urdu", "arabic"):
        text = _DIACRITICS.sub("", text)
        text = "".join(_URDU_VARIANTS.get(ch, ch) for ch in text)
        text = _PUNCT_AR.sub(" ", text)
    elif lang == "hindi":
        text = "".join(_HINDI_VARIANTS.get(ch, ch) for ch in text)
        text = _PUNCT_DEV.sub(" ", text)
    elif lang == "spanish":
        text = text.lower()
        text = _PUNCT.sub(" ", text)
    else:  # english and fallback
        text = text.lower()
        text = _PUNCT.sub(" ", text)

    text = _WS.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------- WER computation

def compute_wer(references, hypotheses):
    """Overall WER via jiwer. Returns (wer, details)."""
    if not references:
        return float("nan"), {}
    correct = 0
    total = 0
    for ref, hyp in zip(references, hypotheses):
        r = ref.split()
        h = hyp.split()
        total += len(r)
        # simple LCS-based correct count
        import difflib
        sm = difflib.SequenceMatcher(a=r, b=h)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                correct += i2 - i1
    wer = 1.0 - correct / total if total else 0.0
    return wer, {"correct": correct, "total_words": total}


def evaluate(metadata_path, hypotheses):
    """Compute per-language and mixed/pure WER.

    hypotheses: dict mapping metadata row id (or path) -> predicted text.
    Returns a dict with overall/per-language/mixed WER and per-utterance rows.
    """
    rows = []
    with open(metadata_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    per_lang = {}
    mixed = {"refs": [], "hyps": []}
    pure = {"refs": [], "hyps": []}
    all_refs, all_hyps = [], []
    details = []

    for row in rows:
        path = row["path"]
        lang = row.get("language", "english").lower()
        is_mixed = str(row.get("is_mixed", "0")).strip() in ("1", "true", "yes")
        ref = normalize_text(row.get("text", ""), lang)
        hyp = normalize_text(hypotheses.get(path, ""), lang)
        wer, det = compute_wer([ref], [hyp])
        details.append({"path": path, "language": lang, "is_mixed": is_mixed,
                        "ref": ref, "hyp": hyp, "wer": wer})

        per_lang.setdefault(lang, {"refs": [], "hyps": []})
        per_lang[lang]["refs"].append(ref)
        per_lang[lang]["hyps"].append(hyp)
        (mixed if is_mixed else pure)["refs"].append(ref)
        (mixed if is_mixed else pure)["hyps"].append(hyp)
        all_refs.append(ref)
        all_hyps.append(hyp)

    result = {"overall": compute_wer(all_refs, all_hyps)[0], "per_language": {},
              "mixed": compute_wer(mixed["refs"], mixed["hyps"])[0],
              "pure": compute_wer(pure["refs"], pure["hyps"])[0],
              "n": len(rows), "details": details}
    for lang, d in per_lang.items():
        result["per_language"][lang] = compute_wer(d["refs"], d["hyps"])[0]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--hypotheses", required=True,
                    help="JSON file: {path: predicted_text} or list of {path, text}")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.hypotheses, encoding="utf-8") as f:
        hyp = json.load(f)
    if isinstance(hyp, list):
        hyp = {h["path"]: h.get("text", h.get("hyp", "")) for h in hyp}

    result = evaluate(args.metadata, hyp)
    summary = {k: v for k, v in result.items() if k != "details"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
