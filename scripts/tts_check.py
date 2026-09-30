"""TTS intelligibility check: synthesize test sentences, transcribe back with STT, compute WER.

For each language, synthesize a few sentences with Piper TTS, transcribe them
with Whisper STT, and compute WER against the source text. Saves:
  - results/tts_check/<lang>/<i>.wav   (synthesized audio)
  - results/tts_check.json              (per-language WER + details)

Usage:
  py scripts/tts_check.py
  py scripts/tts_check.py --stt-model openai/whisper-tiny --limit 3
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval import normalize_text, compute_wer

TEST_SENTENCES = {
    "english": [
        "Hello, thank you for calling Sunrise Family Clinic.",
        "We are open Monday to Friday from 9 AM to 5 PM.",
        "A general consultation costs $50.",
        "I can connect you to a human representative.",
    ],
    "urdu": [
        "السلام علیکم، سنرے فیملی کلینک پر کال کرنے کا شکریہ۔",
        "ہم پیر تا جمعہ صبح 9 سے شام 5 بجے تک کھلے ہیں۔",
        "عمومی معائنے کی قیمت 50 ڈالر ہے۔",
        "میں آپ کو کسی انسانی نمائندے سے جوڑ سکتا ہوں۔",
    ],
    "hindi": [
        "नमस्ते, सनराइज़ फैमिली क्लिनिक पर कॉल करने के लिए धन्यवाद।",
        "हम सोमवार से शुक्रवार सुबह 9 से शाम 5 बजे तक खुले हैं।",
        "सामान्य परामर्श की कीमत 50 डॉलर है।",
        "मैं आपको किसी मानव प्रतिनिधि से जोड़ सकता हूँ।",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt-model", default="openai/whisper-tiny")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/tts_check.json")
    args = ap.parse_args()

    from src.tts import TTS
    from src.pipeline import WhisperSTT

    stt = WhisperSTT(model=args.stt_model)
    out_dir = os.path.join("results", "tts_check")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for lang, sentences in TEST_SENTENCES.items():
        if args.limit:
            sentences = sentences[: args.limit]
        refs, hyps, rows = [], [], []
        tts = TTS(lang)
        for i, sent in enumerate(sentences):
            t0 = time.time()
            wav = tts.synthesize(sent)
            synth_t = time.time() - t0
            fpath = os.path.join(out_dir, lang, f"{i:02d}.wav")
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "wb") as f:
                f.write(wav)
            # transcribe back (array input — no ffmpeg needed)
            import soundfile as sf
            pcm, sr = sf.read(fpath, dtype="float32")
            if pcm.ndim > 1:
                pcm = pcm.mean(axis=1)
            hyp = stt.transcribe_array(pcm, sr)
            ref_n = normalize_text(sent, lang)
            hyp_n = normalize_text(hyp, lang)
            wer, det = compute_wer([ref_n], [hyp_n])
            refs.append(ref_n)
            hyps.append(hyp_n)
            rows.append({"i": i, "ref": sent, "hyp": hyp, "wer": wer,
                         "synth_seconds": round(synth_t, 3)})
            print(f"  [{lang}] {i}: wer={wer:.2f} synth={synth_t:.1f}s")
            print(f"      ref: {sent}")
            print(f"      hyp: {hyp}")
        lang_wer, lang_det = compute_wer(refs, hyps)
        results[lang] = {"wer": lang_wer, "n": len(sentences),
                         "correct": lang_det["correct"], "total_words": lang_det["total_words"],
                         "rows": rows}
        print(f"{lang}: WER={lang_wer:.3f} ({lang_det['correct']}/{lang_det['total_words']} words)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    for lang, r in results.items():
        print(f"  {lang}: WER={r['wer']:.3f}")


if __name__ == "__main__":
    main()
