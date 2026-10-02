"""Automated health check for the call UI: tests the whole call flow with a
synthetic caller (TTS audio played into the pipeline) and confirms a spoken
reply comes back. No browser or microphone needed.

Usage:
  py scripts/check_demo.py                # GPU STT (default)
  py scripts/check_demo.py --cpu          # force CPU STT
  py scripts/check_demo.py --scenario restaurant

Exit code 0 = pass, 1 = fail.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="force CPU STT")
    ap.add_argument("--scenario", default="clinic", choices=["clinic", "restaurant"])
    ap.add_argument("--stt-model", default="openai/whisper-small")
    args = ap.parse_args()

    import numpy as np
    from src.tts import TTS
    from src.pipeline import Pipeline

    print("=== check_demo: automated call-flow health check ===")
    print(f"scenario={args.scenario} stt_model={args.stt_model} cpu={args.cpu}")

    # 1. build the pipeline
    t0 = time.time()
    pipe = Pipeline(stt_model=args.stt_model)
    if args.cpu:
        pipe.stt.device = "cpu"
    print(f"pipeline ready in {time.time()-t0:.1f}s")

    # 2. synthesize a caller utterance (TTS -> audio)
    caller_text = "hello, I want to book an appointment"
    print(f"synthesizing caller audio: {caller_text!r}")
    tts = TTS("english")
    wav = tts.synthesize(caller_text)
    import soundfile as sf
    import io
    pcm, sr = sf.read(io.BytesIO(wav), dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    print(f"caller audio: {len(pcm)/sr:.1f}s at {sr} Hz")

    # 3. run it through the pipeline (VAD -> STT -> brain -> TTS)
    t0 = time.time()
    result = pipe.process_array(pcm)
    dt = time.time() - t0
    print(f"transcript: {result.transcript!r}")
    print(f"intent: {result.intent}")
    print(f"language: {result.detected_language}")
    print(f"reply: {result.reply_text!r}")
    print(f"latency: {dt:.2f}s {result.latencies}")

    # 4. assertions
    ok = True
    if not result.transcript:
        print("FAIL: empty transcript (STT produced nothing)")
        ok = False
    if not result.reply_text:
        print("FAIL: empty reply (brain produced nothing)")
        ok = False
    if not result.reply_audio:
        print("FAIL: no reply audio (TTS produced nothing)")
        ok = False
    # the reply must mention the AI + recording (greeting rule)
    if result.intent == "greeting":
        g = result.reply_text.lower()
        if "ai" not in g:
            print("FAIL: greeting missing 'AI'")
            ok = False
        if "record" not in g:
            print("FAIL: greeting missing 'recording'")
            ok = False

    # 5. a second turn (FAQ) to confirm multi-turn works
    faq_audio = tts.synthesize("what are your opening hours?")
    pcm2, sr2 = sf.read(io.BytesIO(faq_audio), dtype="float32")
    if pcm2.ndim > 1:
        pcm2 = pcm2.mean(axis=1)
    result2 = pipe.process_array(pcm2)
    print(f"turn2 transcript: {result2.transcript!r}")
    print(f"turn2 intent: {result2.intent}")
    if not result2.reply_text:
        print("FAIL: turn2 empty reply")
        ok = False

    print(f"\ncheck_demo: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
