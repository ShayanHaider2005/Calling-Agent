"""Benchmark: per-stage latency, end-to-end delay, peak VRAM and RAM.

Runs the pipeline over a set of scripted turns and measures:
  - per-stage latency (vad, stt, brain, tts) from the latency log
  - end-to-end response delay (total per turn)
  - peak RAM (psutil) and peak VRAM (torch, 0 if CPU-only)

Writes results/benchmark_results.json and prints a summary.
LATENCY_REPORT.md is written separately (by hand / docs task).

Usage:
  py scripts/benchmark.py
  py scripts/benchmark.py --turns 5
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def measure_ram():
    """Return (current_rss_mb, peak_rss_mb) for this process."""
    import psutil
    proc = psutil.Process()
    rss = proc.memory_info().rss / 1024**2
    try:
        peak = proc.memory_info().peak_wset / 1024**2 if hasattr(proc.memory_info(), "peak_wset") else rss
    except Exception:
        peak = rss
    return rss, peak


def measure_vram():
    """Return peak VRAM in MB (0 if no CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return 0.0


def run_brain_benchmark(turns):
    """Text-mode brain-only latency over many turns."""
    from src.brain import Brain
    brain = Brain()
    texts = [
        "hello", "what are your opening hours?", "how much does a blood test cost?",
        "what services do you offer?", "where are you located?",
        "i want to book an appointment", "my name is Test User", "monday", "10 am", "yes",
        "are you a human?", "can you promise me a discount?", "goodbye",
    ]
    lat = []
    for i in range(turns):
        t = texts[i % len(texts)]
        t0 = time.time()
        brain.process(t)
        lat.append(time.time() - t0)
    return lat


def run_full_benchmark(turns, audio_path):
    """Full pipeline (file mode) over a few turns. Returns per-turn latencies."""
    from src.pipeline import Pipeline
    pipe = Pipeline()
    results = []
    for i in range(turns):
        t0 = time.time()
        r = pipe.process_file(audio_path)
        results.append(r.to_dict())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=30, help="brain-only turns")
    ap.add_argument("--full-turns", type=int, default=3, help="full-pipeline turns")
    ap.add_argument("--audio", default=None, help="wav file for full-pipeline benchmark")
    ap.add_argument("--out", default="results/benchmark_results.json")
    args = ap.parse_args()

    print("=== Brain-only (text) benchmark ===")
    ram0, _ = measure_ram()
    t0 = time.time()
    brain_lat = run_brain_benchmark(args.turns)
    brain_time = time.time() - t0
    ram1, peak_ram = measure_ram()
    print(f"  {args.turns} turns in {brain_time:.2f}s")
    print(f"  mean={sum(brain_lat)/len(brain_lat)*1000:.1f}ms  "
          f"max={max(brain_lat)*1000:.1f}ms  min={min(brain_lat)*1000:.1f}ms")
    print(f"  RAM: {ram1:.0f} MB (peak {peak_ram:.0f} MB)")

    full = None
    if args.audio and os.path.exists(args.audio):
        print(f"\n=== Full pipeline benchmark ({args.full_turns} turns, {args.audio}) ===")
        full = run_full_benchmark(args.full_turns, args.audio)
        for r in full:
            print(f"  total={r['total_latency']:.2f}s  {r['latencies']}")
    else:
        print("\n(no --audio given; skipping full-pipeline benchmark)")

    vram = measure_vram()
    print(f"\nPeak VRAM: {vram:.0f} MB (0 = CPU-only torch)")

    out = {
        "brain_only": {
            "turns": args.turns,
            "mean_ms": sum(brain_lat) / len(brain_lat) * 1000,
            "max_ms": max(brain_lat) * 1000,
            "min_ms": min(brain_lat) * 1000,
        },
        "full_pipeline": full,
        "ram_mb": {"start": ram0, "end": ram1, "peak": peak_ram},
        "peak_vram_mb": vram,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
